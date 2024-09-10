import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class DrugAttentionLayer(nn.Module):
    def __init__(self, embed_dim, alpha):
        super(DrugAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.alpha = alpha

        # 定义可学习的注意力向量 a_{\Phi_m}，用于注意力计算
        self.a_phim = nn.Parameter(torch.empty(size=(2 * embed_dim, 1)))
        nn.init.xavier_uniform_(self.a_phim.data, gain=1.414)

        # LeakyReLU 用于激活线性变换的结果
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, drug_embeddings, drug_relationships):
        N = drug_embeddings.size()[0]  # 药物数量

        # 创建一个字典，记录每个药物连接的所有药物
        neighbors = defaultdict(set)
        for (i, j) in drug_relationships:
            neighbors[i].add(j)
            neighbors[j].add(i)

        h_prime = drug_embeddings  # 初始化特征聚合结果
        h_prime = h_prime.clone()
        # 遍历每个药物，计算其连接药物的注意力并进行聚合
        for drug_i, drug_js in neighbors.items():
            attention_scores = []
            drug_i_emb = drug_embeddings[drug_i]  # 当前药物的嵌入

            # 计算 drug_i 与所有连接药物 drug_js 的注意力权重
            for drug_j in drug_js:
                drug_j_emb = drug_embeddings[drug_j]
                e_ij = torch.cat([drug_i_emb, drug_j_emb], dim=0)  # 拼接 drug_i 和 drug_j 的嵌入
                attention_ij = self.leakyrelu(torch.matmul(e_ij, self.a_phim).squeeze())
                attention_scores.append(attention_ij)

            # 对 attention_scores 进行 softmax 归一化
            attention_scores = torch.tensor(attention_scores)
            attention_scores = F.softmax(attention_scores, dim=0)

            h_prime = h_prime.clone()
            # 使用注意力权重对连接药物的嵌入进行加权聚合
            for idx, drug_j in enumerate(drug_js):
                h_prime[drug_i] = h_prime[drug_i] + attention_scores[idx] * drug_embeddings[drug_j]



        return h_prime
