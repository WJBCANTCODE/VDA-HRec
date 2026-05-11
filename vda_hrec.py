# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender


class GlobalHypergraphRefiner(nn.Module):
    def __init__(self, n_layers):
        super(GlobalHypergraphRefiner, self).__init__()
        self.n_layers = n_layers
    def forward(self, incidence_matrix_i, incidence_matrix_u, base_embeddings):
        refined_item_repr = base_embeddings
        for _ in range(self.n_layers):
            latent_manifold = torch.mm(incidence_matrix_i.T, refined_item_repr)
            refined_item_repr = torch.mm(incidence_matrix_i, latent_manifold)
            refined_user_repr = torch.mm(incidence_matrix_u, latent_manifold)
        return refined_user_repr, refined_item_repr
class HyperConv(nn.Module):
    def __init__(self, layers, emb_size, k_hops=2):
        super().__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.k_hops = k_hops
        self.var_weight = 1e-5
    def compute_feature_variance(self, x):
        global_mean = x.mean(dim=0, keepdim=True)  # [1, D]
        global_var = x.var(dim=0, keepdim=True, unbiased=True)  # [1, D]
        node_dim_var = (x - global_mean) **2
        global_var = torch.clamp(global_var, min=1e-8) 
        node_dim_var_norm = node_dim_var / global_var
        return node_dim_var_norm
    def forward(self, adjacency, embedding):
        item_embeddings = embedding  # [N, D]
        final = [item_embeddings]
        adjacency = adjacency.to(embedding.device)
        var_features = self.compute_feature_variance(item_embeddings)
        for i in range(self.layers):
            for hop in range(self.k_hops):
                item_embeddings = torch.sparse.mm(adjacency, item_embeddings)
                item_embeddings = item_embeddings + self.var_weight * var_features     
            final.append(item_embeddings)
        item_embeddings = torch.sum(torch.stack(final), 0) / (self.layers + 1)  
        return item_embeddings

class VDA_HRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(VDA_HRec, self).__init__(config, dataset)
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.cf_model = config['cf_model']
        self.n_mm_layer = config['n_mm_layer']
        self.n_ui_layers = config['n_ui_layers']
        self.n_hyper_layer = config['n_hyper_layer']
        self.hyper_num = config['hyper_num']
        self.keep_rate = config['keep_rate']
        self.alpha = config['alpha']
        self.cl_weight = config['cl_weight']
        self.reg_weight = config['reg_weight']
        self.tau = 0.2
        self.n_nodes = self.n_users + self.n_items
        self.gh_refiner = GlobalHypergraphRefiner(self.n_hyper_layer)
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.item_adj_0 = self.interaction_matrix.T * self.interaction_matrix
        self.item_adj = self.scipy_matrix_to_sparse_tenser(self.item_adj_0, torch.Size((self.n_items, self.n_items)))
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.drop = nn.Dropout(p=1-self.keep_rate)
        self.item_image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim, bias=False)
        self.item_text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim, bias=False)
        nn.init.xavier_uniform_(self.item_image_trs.weight)
        nn.init.xavier_uniform_(self.item_text_trs.weight)
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            #self.item_image_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1],self.feat_embed_dim)))
            self.v_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.hyper_num)))
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            #self.item_text_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))
            self.t_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.hyper_num)))
        
        self.collaborative_hyperconv = HyperConv(
            layers=self.n_ui_layers,
            emb_size=self.embedding_dim,
            k_hops=2 # k_hops=2 
        )    
        self.knn_k = 10 
        self.knn_adj = None     
        knn_adj_list = []
        if self.v_feat is not None:
            v_feat_tensor = self.image_embedding.weight.detach()
            v_knn = self.get_knn_adj_mat(v_feat_tensor, k=self.knn_k)
            knn_adj_list.append(v_knn)
        if self.t_feat is not None:
            t_feat_tensor = self.text_embedding.weight.detach()
            t_knn = self.get_knn_adj_mat(t_feat_tensor, k=self.knn_k)
            knn_adj_list.append(t_knn)
        if len(knn_adj_list) > 0:
            if len(knn_adj_list) == 1:
                self.knn_adj = knn_adj_list[0]
            else:
                self.knn_adj = (knn_adj_list[0] + knn_adj_list[1]).coalesce()
                self.knn_adj = torch.sparse.FloatTensor(
                    self.knn_adj.indices(), 
                    self.knn_adj.values() *0.5, 
                    self.knn_adj.size()
                ).to(self.device)
        else:
            self.knn_adj = self.item_adj

    def get_knn_adj_mat(self, mm_embeddings, k=10, batch_size=8192):
        embeds = F.normalize(mm_embeddings, p=2, dim=1).to(self.device)
        num_items = embeds.shape[0]
        indices_list = []
        values_list = []
        for i in range(0, num_items, batch_size):
            end = min(i + batch_size, num_items)
            batch_emb = embeds[i:end]
            sim = torch.mm(batch_emb, embeds.t())
            vals, inds = torch.topk(sim, k, dim=1)
            rows = torch.arange(i, end, device=self.device).view(-1, 1).expand(-1, k).flatten()
            cols = inds.flatten()
            indices_list.append(torch.stack([rows, cols]))
            values_list.append(vals.flatten())
        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size([num_items, num_items])).to(self.device)
        return self.normalize_graph(adj)

    def normalize_graph(self, adj):
        adj = adj.coalesce()
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0
        indices = adj.indices()
        values = adj.values()
        row_indices = indices[0]
        col_indices = indices[1]
        norm_values = values * degree_inv_sqrt[row_indices] * degree_inv_sqrt[col_indices]
        return torch.sparse.FloatTensor(indices, norm_values, adj.size()).to(self.device)

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        if not isinstance(matrix, sp.coo_matrix):
            matrix = matrix.tocoo()
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        sparse_tensor = torch.sparse.FloatTensor(i, data, shape).coalesce()

        return sparse_tensor.to(self.device)
    
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = inter_M.transpose()
        for u, i in zip(inter_M.row, inter_M.col):
            A[u, i + self.n_users] = 1
        for i, u in zip(inter_M_t.row, inter_M_t.col):
            A[i + self.n_users, u] = 1
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))
    def cge(self):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        cge_embs = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            cge_embs += [ego_embeddings]
        cge_embs = torch.stack(cge_embs, dim=1)
        cge_embs = cge_embs.mean(dim=1, keepdim=False)
        ego_collab = self.collaborative_hyperconv(self.norm_adj, cge_embs)
        return ego_collab
    def mge_v(self):
        item_ids = torch.arange(self.n_items, device=self.device) 
        item_id_feats = self.item_id_embedding(item_ids)  
        item_feats_v =  self.item_image_trs(self.image_embedding.weight)
        user_feats = torch.sparse.mm(self.adj, item_feats_v) * self.num_inters[:self.n_users]#(19445,1)self.adj(19445,7050)
        mge_feats = torch.concat([user_feats, item_feats_v], dim=0)
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)
        return mge_feats
    def mge_t(self):
        item_feats_t = self.item_text_trs(self.text_embedding.weight)
        user_feats = torch.sparse.mm(self.adj, item_feats_t) * self.num_inters[:self.n_users]
        mge_feats = torch.concat([user_feats, item_feats_t], dim=0)
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)
        return mge_feats#(26945,64)
    def i_i_grah(self):
        i_i_v_feat = self.item_image_trs(self.image_embedding.weight)
        i_i_t_feat = self.item_text_trs(self.text_embedding.weight)
        i_i_feat_0 = i_i_v_feat + i_i_t_feat
        graph_to_use = 0.5* self.knn_adj + self.item_adj
        for _ in range(self.n_mm_layer):
            i_i_feat = torch.sparse.mm(graph_to_use, i_i_feat_0) + i_i_feat_0      
        return i_i_feat
    def forward(self):
        v_feats = self.mge_v()
        t_feats = self.mge_t()
        if self.v_feat is not None:
            iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)#(模态维度，hype_num)
            uv_hyper = torch.mm(self.adj, iv_hyper)
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)
        if self.t_feat is not None:
            it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)
            ut_hyper = torch.mm(self.adj, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            ut_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)
        cge_embs = self.cge()
        i_i_feat = self.i_i_grah()
        if self.v_feat is not None and self.t_feat is not None:
            mge_embs = F.normalize(v_feats) + F.normalize(t_feats)
            lge_embs = cge_embs + mge_embs 
            uv_hyper_embs, iv_hyper_embs = self.gh_refiner(self.drop(iv_hyper), self.drop(uv_hyper), cge_embs[self.n_users:])
            iv_hyper_embs = torch.matmul(self.item_adj, iv_hyper_embs) + i_i_feat
            ut_hyper_embs, it_hyper_embs = self.gh_refiner(self.drop(it_hyper), self.drop(ut_hyper), cge_embs[self.n_users:])
            iv_hyper_embs = torch.matmul(self.item_adj, it_hyper_embs) + i_i_feat
            av_hyper_embs = torch.concat([uv_hyper_embs, iv_hyper_embs], dim=0)
            at_hyper_embs = torch.concat([ut_hyper_embs, it_hyper_embs], dim=0)
            ghe_embs = av_hyper_embs + at_hyper_embs
            all_embs = lge_embs + self.alpha * F.normalize(ghe_embs)
        else:
            all_embs = cge_embs
        intermediate_embs = {
            'cge': cge_embs,
            'v': v_feats,
            't': t_feats,
            'mge': mge_embs,
            'i_i_feat': i_i_feat
        }
        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)

        return u_embs, i_embs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs],  intermediate_embs
    def fit_Gaussian_dis(self, *embs):
        results = []
        for emb in embs:
            current_var = torch.var(emb)
            current_mean = torch.mean(emb)
            results.append((current_var, current_mean))
        return results

    def calculate_align_loss(self, embs_dict):
        cge_embs = embs_dict['cge']
        v_feats = embs_dict['v']
        t_feats = embs_dict['t']
        mge_embs = embs_dict['mge']
        results = self.fit_Gaussian_dis(cge_embs, v_feats, t_feats, mge_embs)
        align_loss = 0.0
        embs_list = list(results)
        for i in range(len(embs_list)):
            for j in range(i + 1, len(embs_list)):
                var_i, mean_i = embs_list[i]
                var_j, mean_j = embs_list[j]
                diff = torch.abs(var_i - var_j) + torch.abs(mean_i - mean_j)
                align_loss += diff.mean() 
        return align_loss      
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss
    
    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss
    
    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def calculate_loss(self, interaction):
        ua_embeddings, ia_embeddings, hyper_embeddings, intermediate_embs= self.forward()
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        i_i_embs = intermediate_embs['i_i_feat']
        pos_i_final = ia_embeddings[pos_items]
        pos_i_i_feat = i_i_embs[pos_items]    
        neg_i_i_feat = i_i_embs[neg_items]
        icl_loss_1 = self.ssl_triple_loss( pos_i_i_feat, neg_i_i_feat, i_i_embs)
        [uv_embs, iv_embs, ut_embs, it_embs] = hyper_embeddings
        batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs) + self.ssl_triple_loss(iv_embs[pos_items], it_embs[pos_items], it_embs)
        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        loss = batch_bpr_loss + \
            self.cl_weight * batch_hcl_loss + \
            self.reg_weight * batch_reg_loss + \
            0.1*self.cl_weight*icl_loss_1
        return loss
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs,  _ ,_= self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores