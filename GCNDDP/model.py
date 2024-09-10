import random

import torch
import torch.nn as nn

from DrugAttentionLayer import DrugAttentionLayer
from GeneAttentionLayer import GeneAttentionLayer
from utils import sparse_dropout, spmm
import torch.nn.functional as F
import numpy as np

class GCNDDP(nn.Module):
    def __init__(self, n_u, n_i, d, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout,
                 device):
        super(GCNDDP, self).__init__()
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



        self.attention_layer1 = DrugAttentionLayer(256, 0.2)
        self.attention_layer2 = GeneAttentionLayer(256, 0.2)



    def forward(self, uids, iids, pos, neg, sampled_gene_gene_relationships, sampled_drug_drug_relationships,  test=False):
        if test == True:  # testing phase
            u_emb = self.E_g_GNN[uids]
            i_emb = self.E_d_GNN[iids]

            # 合并并通过 MLP
            u_i_concat = torch.cat([u_emb, i_emb], dim=-1).unsqueeze(0)
            pred = self.concat_mlp(u_i_concat).squeeze(-1)

            return pred, self.E_g_GNN, self.E_d_GNN
        else:  # training phase
            E_d_GNN_0 = self.attention_layer2(self.E_d_GNN_0, sampled_drug_drug_relationships)
            E_d_GNN_0 = 0.1 * E_d_GNN_0 + self.E_d_GNN_0
            # E_d_GNN_0 = torch.cat([E_d_GNN_0, self.E_d_GNN_0], dim=-1)

            # E_d_GNN_0 = self.concat_mlp1(E_d_GNN_0)

            # E_d_GNN_0 = self.E_d_GNN_0

            # 假设 gene_gene_relationships 是一个列表或张量，其中包含基因之间的关系
            # 随机选取 1/10 的基因关系
            # num_relationships = len(gene_gene_relationships)  # 计算总关系数
            # sample_size = max(1, num_relationships // 25)  # 确保至少选取1个关系
            #
            # # 从 gene_gene_relationships 中随机选取 sample_size 数量的关系
            # sampled_gene_gene_relationships = random.sample(gene_gene_relationships, sample_size)
            #
            E_g_GNN_0 = self.attention_layer2(self.E_g_GNN_0, sampled_gene_gene_relationships)
            E_g_GNN_0 = 0.1 * E_g_GNN_0 +  self.E_g_GNN_0



            self.E_g_GNN = torch.spmm(sparse_dropout(self.adj_norm, self.dropout), E_d_GNN_0)

            self.E_d_GNN = torch.spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), E_g_GNN_0)




            #交叉熵损失
            u_emb = self.E_g_GNN[uids]  # 4096,1024 #基因
            pos_emb = self.E_d_GNN[pos]  # 4096,1024 # 药物
            neg_emb = self.E_d_GNN[neg]  # 4096,1024 # 药物

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
            loss_r = loss_pos + loss_neg
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean() + loss_r  # 差值越大越好



            loss_s = torch.tensor(0.0)


            # reg loss
            # loss_reg 是正则化损失，用于避免模型过拟合，通过对所有参数施加 L2 正则化来实现。
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2


            # total loss
            loss = loss_reg + self.lambda_1 * loss_s  + loss_r

            return loss, loss_r, self.lambda_1 * loss_s





