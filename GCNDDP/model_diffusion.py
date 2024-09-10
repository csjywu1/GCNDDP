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

            # diffusion的损失
            a = u_emb  # gene
            b = pos_emb  # drug_pos
            c = neg_emb  # drug_neg

            pos_score = torch.sum(a * b, dim=1)
            pos_score = -torch.mean(F.sigmoid(pos_score))  # F.logsigmoid(pos))

            neg_score = torch.sum(a * c, dim=1)
            neg_score = -torch.mean(F.sigmoid(-neg_score))  # F.logsigmoid(-neg)

            def train_diffusion():
                nei_output = self.E_d_GNN[pos].detach()  # 邻居节点的特征
                n_output = self.E_d_GNN[uids].detach()  # 复制成一样的维度

                for epoch in range(3):
                    self.d_optimizer.zero_grad()
                    dif_loss = self.diffusion(nei_output, n_output, self.device)
                    dif_loss.backward(retain_graph=True)
                    self.d_optimizer.step()
                    # print("Diffusion_Loss:", dif_loss.item())
                return

            train_diffusion()

            h_syn = self.diffusion.sample(a.shape, a)  # 扩散过程后，生成合成负样本 (h_syn)，引入基于扩散的额外负样本。
            neg_list = []
            w = [0, 1, 0.9, 0.8, 0.7]

            for i in range(len(h_syn)):
                syn_neg = torch.sum(a * h_syn[i], dim=1)  # 每个合成负损失类似于主要负损失，通过 a 和生成的合成负样本的点积计算，再应用 Sigmoid 函数并求平均。
                syn_neg = -torch.mean(F.sigmoid(-syn_neg))  # F.logsigmoid(-syn_neg)
                neg_list.append(w[i] * syn_neg)  # 最终负损失作为 neg_list 中各项损失的加权和计算得出。

            neg_list.append(neg_score)
            sam = [1 for i in w if i != 0]
            neg_score = sum(neg_list) / (sum(sam) + 1)

            loss_s = (pos_score + neg_score) / 2  # 最终损失为正损失和负损失的平均值 (loss = (pos + neg)/2)


            # reg loss
            # loss_reg 是正则化损失，用于避免模型过拟合，通过对所有参数施加 L2 正则化来实现。
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2


            # total loss
            loss = loss_reg + self.lambda_1 * loss_s  + loss_r

            return loss, loss_r, loss_s





