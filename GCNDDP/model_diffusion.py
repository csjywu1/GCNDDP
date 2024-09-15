import random

import torch
import torch.nn as nn

from DrugAttentionLayer import DrugAttentionLayer
from GeneAttentionLayer import GeneAttentionLayer
from model_cond import Diffusion_Cond
from utils import sparse_dropout, spmm
import torch.nn.functional as F
import numpy as np

class GCNDDP_diffusion(nn.Module):
    def __init__(self, n_u, n_i, d, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout,
                 device):
        super(GCNDDP_diffusion, self).__init__()
        # 初始化 nn层
        self.E_g_GNN_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))  # 3227， 1024
        self.E_d_GNN_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))  # 10690， 1024


        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_g_GNN_list = [None] * (l + 1)
        self.E_d_GNN_list = [None] * (l + 1)
        self.E_g_GNN_list[0] = self.E_g_GNN_0
        self.E_d_GNN_list[0] = self.E_d_GNN_0


        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)

        self.E_g_GNN = None
        self.E_d_GNN = None

        self.device = device

        self.concat_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1)  # 输出维度为1，用于计算分数
        )

        self.concat_mlp1 = nn.Sequential(
            nn.Linear(2*d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            # nn.ReLU(),
            # nn.Linear(d, 1)  # 输出维度为1，用于计算分数
        )

        self.diffusion = Diffusion_Cond(256, 256, 256)

        self.d_optimizer = torch.optim.Adam(self.diffusion.parameters(), \
                                       lr=0.01, weight_decay=0.99)


        self.attention_layer1 = DrugAttentionLayer(256, 0.2)
        self.attention_layer2 = GeneAttentionLayer(256, 0.2)



    def forward(self, uids, iids, pos, neg, sampled_gene_gene_relationships, sampled_drug_drug_relationships,  test=False):
         if test == True:  # testing phase
            i_emb = self.E_g_GNN[uids]  # gene
            u_emb = self.E_d_GNN[iids]  # drug

            # 合并并通过 MLP
            u_i_concat = torch.cat([u_emb, i_emb], dim=-1).unsqueeze(0)

            h_syn = self.diffusion.sample(u_emb.shape, u_emb)  # 扩散过程后，生成合成负样本 (h_syn)，引入基于扩散的额外负样本。 # 这里会生成5个负样本
            neg_list = []
            w = [1, 0.6, 0.3, 0.1]  # 这是权重，不同hard的样本有不同的权重

            # 假设 h_syn 是一个列表，每个元素是一个 embedding

            combined_neg_embeddings = torch.zeros_like(h_syn[0])  # 初始化一个和 h_syn[0] 形状相同的全零张量，用于累加加权的负样本嵌入

            for i in range(len(h_syn)):  # 能否生成一个固定sample的列表。现在太慢了
                weighted_syn_neg = w[i] * h_syn[i]  # 对每个 h_syn[i] 进行加权
                combined_neg_embeddings += weighted_syn_neg

            u_i_concat1 = torch.cat([u_emb, i_emb], dim=-1).unsqueeze(0)

            u_i_concat2 = torch.cat([u_emb, combined_neg_embeddings], dim=-1).unsqueeze(0)

            # self.concat_mlp(u_i_concat).squeeze(-1)+
            # pred =  (self.concat_mlp1(u_i_concat1) - self.concat_mlp1 (u_i_concat2))
            pred = (self.concat_mlp1(u_i_concat1) - self.concat_mlp1(u_i_concat2)).sigmoid() + self.concat_mlp1(u_i_concat1).sigmoid()

            return pred, self.E_g_GNN, self.E_d_GNN
        else:  # training phase
            E_d_GNN_0 = self.attention_layer2(self.E_d_GNN_0, sampled_drug_drug_relationships)
            E_d_GNN_0 = E_d_GNN_0 + self.E_d_GNN_0

            #
            E_g_GNN_0 = self.attention_layer2(self.E_g_GNN_0, sampled_gene_gene_relationships)
            E_g_GNN_0 = E_g_GNN_0 + self.E_g_GNN_0

            self.E_d_GNN = torch.spmm(sparse_dropout(self.adj_norm, self.dropout), E_g_GNN_0)

            self.E_g_GNN = torch.spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), E_d_GNN_0)

            # 交叉熵损失
            u_emb = self.E_d_GNN[uids]  # 4096,1024 #药物
            pos_emb = self.E_g_GNN[pos]  # 4096,1024 # 基因
            neg_emb = self.E_g_GNN[neg]  # 4096,1024 # 基因

            # 正样本和负样本的得分
            u_pos_concat = torch.cat([u_emb, pos_emb], dim=-1)
            u_neg_concat = torch.cat([u_emb, neg_emb], dim=-1)

            # 通过MLP进行降维
            pos_scores = self.concat_mlp(u_pos_concat).squeeze(-1)
            neg_scores = self.concat_mlp(u_neg_concat).squeeze(-1)

            # 创建标签
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)

            # 计算交叉熵损失
            loss_pos = F.binary_cross_entropy_with_logits(pos_scores, pos_labels)
            loss_neg = F.binary_cross_entropy_with_logits(neg_scores, neg_labels)

            # 总的交叉熵损失
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean() + loss_pos + loss_neg

            # loss_r = loss_r  # 差值越大越好

            #

            # diffusion的损失
            a = u_emb  # drug
            b = pos_emb  # gene_pos
            c = neg_emb  # gene_neg

            # pos_score = torch.sum(a * b, dim=1)
            # pos_score = torch.mean(F.sigmoid(pos_score))  # F.logsigmoid(pos))

            # 假设您已经定义了一个 MLP 网络 self.mlp 用于计算相似度

            # neg_score = self.concat_mlp(torch.cat((a, c), dim=1))  # 用a和c的concat通过mlp计算负样本的分数
            # neg_score = torch.mean(F.sigmoid(neg_score))  # 负样本的分数通过sigmoid激活后取均值

            def train_diffusion():
                nei_output = self.E_g_GNN[pos].detach()  # 邻居节点的特征
                n_output = self.E_d_GNN[uids].detach()  # 复制成一样的维度

                for epoch in range(100):
                    self.d_optimizer.zero_grad()
                    dif_loss = self.diffusion(nei_output, n_output, self.device)  # 扩散计算损失
                    dif_loss.backward(retain_graph=True)
                    self.d_optimizer.step()

                return

            if self.i == 1:
                train_diffusion()
                self.i = 2

            h_syn = self.diffusion.sample(a.shape, a)  # 扩散过程后，生成合成负样本 (h_syn)，引入基于扩散的额外负样本。 # 这里会生成5个负样本
            neg_list = []
            w = [1, 0.9, 0.8, 0.7]  # 这是权重，不同hard的样本有不同的权重

            # 假设 h_syn 是一个列表，每个元素是一个 embedding

            combined_neg_embeddings = torch.zeros_like(h_syn[0])  # 初始化一个和 h_syn[0] 形状相同的全零张量，用于累加加权的负样本嵌入

            for i in range(len(h_syn)):
                weighted_syn_neg = w[i] * h_syn[i]  # 对每个 h_syn[i] 进行加权
                combined_neg_embeddings += weighted_syn_neg  # 这里生成的是gene的负样本 # 将加权后的嵌入逐步累加

            # neg_list.append(neg_score)  # 将初始的neg_score加入到neg_list
            # sam = [1 for i in w if i != 0]  # 计算有效权重的数量
            # neg_score = sum(neg_list) / (sum(sam) + 1)  # 负样本分数的加权平均

            # combined_neg_embeddings =  torch.cat([neg_emb, combined_neg_embeddings], dim = -1)
            #
            # combined_neg_embeddings = self.concat_mlp2(combined_neg_embeddings)

            neg_concat = torch.cat([a, combined_neg_embeddings], dim=-1)
            syn_neg = self.concat_mlp1(neg_concat).squeeze(-1)

            pos_concat = torch.cat([a, b], dim=-1)
            # 使用交叉熵损失函数计算最终的负样本损失
            syn_pos = self.concat_mlp1(pos_concat).squeeze(-1)

            loss_s1 = F.binary_cross_entropy_with_logits(syn_neg, torch.zeros_like(neg_labels))  # 用交叉熵计算负样本的损失
            #
            loss_s2 = F.binary_cross_entropy_with_logits(syn_pos, torch.ones_like(
                neg_labels)) + F.binary_cross_entropy_with_logits(neg_scores, neg_labels)
            # 用交叉熵计算负样本的损失

            loss_s = -(syn_pos - syn_neg).sigmoid().log().mean() + loss_s1 + loss_s2

            # .sigmoid().log().mean() + loss_s1 + loss_s2

            # + loss_s1 + loss_s2

            def compute_consistency_loss(neg_emb, combined_neg_embeddings, loss_type='mse'):
                """
                Computes the consistency loss between neg_emb and combined_neg_embeddings.

                Args:
                    neg_emb (Tensor): The negative embeddings (B, D) where B is batch size and D is the embedding dimension.
                    combined_neg_embeddings (Tensor): The combined negative embeddings (B, D).
                    loss_type (str): Type of loss to use ('mse', 'cosine', 'kl').

                Returns:
                    Tensor: Consistency loss.
                """
                if loss_type == 'mse':
                    # Mean Squared Error Loss
                    consistency_loss = F.mse_loss(neg_emb, combined_neg_embeddings)

                elif loss_type == 'cosine':
                    # Cosine Similarity Loss (1 - cosine similarity)
                    cosine_sim = F.cosine_similarity(neg_emb, combined_neg_embeddings, dim=-1)
                    consistency_loss = 1 - cosine_sim.mean()

                elif loss_type == 'kl':
                    # KL Divergence Loss
                    neg_emb_log_softmax = F.log_softmax(neg_emb, dim=-1)
                    combined_neg_embeddings_softmax = F.softmax(combined_neg_embeddings, dim=-1)
                    consistency_loss = F.kl_div(neg_emb_log_softmax, combined_neg_embeddings_softmax,
                                                reduction='batchmean')

                else:
                    raise ValueError("Invalid loss_type. Choose from ['mse', 'cosine', 'kl'].")

                return consistency_loss

            consistency_loss = compute_consistency_loss(neg_scores, syn_neg, loss_type='kl')

            # loss_r =  + loss_r  # 差值越大越好

            # F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + \

            # # reg loss
            # # loss_reg 是正则化损失，用于避免模型过拟合，通过对所有参数施加 L2 正则化来实现。
            # loss_reg = 0
            # for param in self.parameters():
            #     loss_reg += param.norm(2).square()
            # loss_reg *= self.lambda_2

            # total loss
            # loss =   loss_r  + 100 *loss_s
            loss = loss_s
            # + consistency_loss +loss_s
            # loss_r +
            # + loss_r
            return loss, loss_r, loss_s, consistency_loss






