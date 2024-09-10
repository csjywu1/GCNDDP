
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
#
# class GeneAttentionLayer(nn.Module):
#     def __init__(self, embed_dim, alpha):
#         super(GeneAttentionLayer, self).__init__()
#         self.embed_dim = embed_dim
#         self.alpha = alpha
#
#         # Learnable attention vector a_phim for attention computation
#         self.a_phim = nn.Parameter(torch.empty(size=(2 * embed_dim, 1)))
#         nn.init.xavier_uniform_(self.a_phim.data, gain=1.414)
#
#         # LeakyReLU activation for the linear transformations
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, drug_embeddings, drug_relationships):
#         N = drug_embeddings.size(0)  # Number of drugs
#
#         # Create a dictionary to store neighbors for each drug
#         neighbors = defaultdict(list)
#         for (i, j) in drug_relationships:
#             neighbors[i].append(j)
#             neighbors[j].append(i)
#
#         # Initialize the output tensor
#         h_prime = drug_embeddings.clone()
#
#         # Vectorize attention score calculation
#         for drug_i, drug_js in neighbors.items():
#             drug_i_emb = drug_embeddings[drug_i].unsqueeze(0)  # Shape: (1, embed_dim)
#             drug_j_embs = drug_embeddings[drug_js]  # Shape: (num_neighbors, embed_dim)
#
#             # Concatenate drug_i with all its neighbors (drug_js) embeddings
#             drug_i_repeat = drug_i_emb.repeat(len(drug_js), 1)  # Repeat drug_i embedding for each neighbor
#             concat_emb = torch.cat([drug_i_repeat, drug_j_embs], dim=1)  # Shape: (num_neighbors, 2 * embed_dim)
#
#             # Calculate attention scores for all neighbors in one go
#             attention_scores = self.leakyrelu(torch.matmul(concat_emb, self.a_phim).squeeze())  # Shape: (num_neighbors)
#
#             # Apply softmax to normalize attention scores
#             attention_scores = F.softmax(attention_scores, dim=0)  # Shape: (num_neighbors)
#
#             # Aggregate neighbor embeddings based on attention scores
#             weighted_sum = torch.matmul(attention_scores.unsqueeze(0), drug_j_embs)  # Shape: (1, embed_dim)
#
#             # Update the embedding for drug_i
#             h_prime[drug_i] += weighted_sum.squeeze()
#
#         return h_prime


class GeneAttentionLayer(nn.Module):
    def __init__(self, embed_dim, alpha):
        super(GeneAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.alpha = alpha

        # LeakyReLU activation for the attention score
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, drug_embeddings, drug_relationships):
        device = drug_embeddings.device
        N = drug_embeddings.size(0)  # Number of drugs

        # Create edge index tensor
        edge_index = torch.tensor(drug_relationships, dtype=torch.long, device=device).t().contiguous()

        # Calculate attention scores for all edges at once
        drug_i = drug_embeddings[edge_index[0]]
        drug_j = drug_embeddings[edge_index[1]]
        attention_scores = self.leakyrelu(torch.sum(drug_i * drug_j, dim=1))

        # Normalize attention scores using softmax
        attention_scores = F.softmax(attention_scores, dim=0)

        # Weighted sum of neighbor embeddings
        weighted_sum = drug_j * attention_scores.unsqueeze(1)

        # Aggregate updates for each node
        h_prime = scatter_add(weighted_sum, edge_index[0], dim=0, dim_size=N)

        # Add the aggregated updates to the original embeddings
        h_prime += drug_embeddings

        return h_prime

