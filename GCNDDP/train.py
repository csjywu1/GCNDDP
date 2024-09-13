import itertools
import random
from collections import defaultdict

import numpy as np
import torch
import pickle
from scipy.sparse import coo_matrix
from model import GCNDDP
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor, mm_auc
from tqdm import tqdm
import torch.utils.data as data
from utils import TrnData
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc as sklearn_auc, \
    precision_recall_curve, recall_score, precision_score, f1_score, auc
from sklearn import metrics as mt
import torch.nn.functional as F
from utils import get_syn_sim
from sklearn.decomposition import PCA
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    # parser.add_argument('--batch', default=256, type=int, help='batch size')  # 256
    parser.add_argument('--inter_batch', default=600, type=int, help='batch size')  # 4096 37514  600
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--lambda1', default=1e-3, type=float, help='weight of cl loss')  # 0.05
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs') #50 10
    parser.add_argument('--d', default=256, type=int, help='embedding size')  # 512 0.886
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')  # 2
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.1, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.8, type=float, help='temperature in cl loss')  # 0.8
    parser.add_argument('--lambda2', default=1e-5, type=float, help='l2 reg weight')  # 1e-5
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')
    return parser.parse_args()

args = parse_args()

device = 'cuda:' + args.cuda
# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
# batch_gene = args.batch
inter_batch = args.inter_batch
epoch_no = args.epoch
# max_samp = 50#40
lambda_1 = args.lambda1

lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q

def generate_negative_samples(pos_samples, num_nodes, num_epochs, num_nodes1):
    neg_samples = {i: [[] for _ in range(num_epochs)] for i in range(num_nodes)}
    for i in range(num_nodes):
        for epoch in range(num_epochs):
            while True:
                j = np.random.randint(0, num_nodes1)
                if (i, j) not in pos_samples:
                    neg_samples[i][epoch].append(j)
                    break
    return neg_samples

def generate_single_negative_samples(train_neg_samples, pos_samples, num_nodes, num_nodes1):
    neg_samples = {i: [] for i in range(num_nodes)}
    for i in range(num_nodes):
        while len(neg_samples[i]) < 1:
            j = np.random.randint(0, num_nodes1)
            if (i, j) not in pos_samples and all(j not in train_neg_samples[i][epoch] for epoch in range(len(train_neg_samples[i]))):
                neg_samples[i].append(j)
    return neg_samples

def load_data():
    with open('data/train_mat_fold0', 'rb') as f:
        train = pickle.load(f)
    with open('data/test_mat_fold0', 'rb') as f:
        test = pickle.load(f)

    # 将数据类型转换为 float64
    train = train.astype(np.float64)
    test = test.astype(np.float64)


    # 转置矩阵，使药物为行，基因为列
    train = train.T
    test = test.T

    return train, test



def preprocess_data(train, test, num_epochs):
    train_coo = coo_matrix(train)
    test_coo = coo_matrix(test)

    train_pos_samples = set(zip(train_coo.row, train_coo.col))
    test_pos_samples = set(zip(test_coo.row, test_coo.col))
    combined_pos_samples = train_pos_samples | test_pos_samples

    num_nodes = train.shape[0]
    num_nodes1 = train.shape[1]
    train_neg_samples = generate_negative_samples(train_pos_samples, num_nodes, num_epochs, num_nodes1)
    test_neg_samples = generate_single_negative_samples(train_neg_samples, combined_pos_samples, num_nodes, num_nodes1)

    train_pos_samples = list(train_pos_samples)
    test_pos_samples = list(test_pos_samples)

    return train_coo, test_coo, train_pos_samples, test_pos_samples, train_neg_samples, test_neg_samples



def train_model(train_loader, model, optimizer, train_neg_samples, num_epochs, train_data):
    loss_list = []
    loss_r_list = []
    loss_s_list = []
    best_loss = float('inf')
    best_model_path = 'best_model.pth'

    import random

    def extract_gene_gene_relationships(drugs, genes, max_neighbors=5):
        # Create a dictionary to store the genes associated with each drug
        drug_to_genes = {}

        # Populate the drug_to_genes dictionary
        for drug, gene in zip(drugs, genes):
            if drug not in drug_to_genes:
                drug_to_genes[drug] = set()
            drug_to_genes[drug].add(gene)

        # Extract gene-gene relationships
        gene_gene_relationships = set()

        # Iterate over drugs and their associated genes
        for genes_set in drug_to_genes.values():
            if len(genes_set) > 1:
                # Limit to max_neighbors if more than max_neighbors genes
                gene_list = sorted(genes_set)
                if len(gene_list) > max_neighbors:
                    gene_list = random.sample(gene_list, max_neighbors)
                elif len(gene_list) < max_neighbors:
                    # Replicate neighbors to reach max_neighbors
                    gene_list += random.choices(gene_list, k=max_neighbors - len(gene_list))

                # Create all gene pairs without calling itertools on each iteration
                gene_gene_relationships.update(
                    (gene_list[i], gene_list[j])
                    for i in range(len(gene_list))
                    for j in range(i + 1, len(gene_list))
                )

        return list(gene_gene_relationships)

    # Assuming train_data.rows contains drugs and train_data.cols contains genes
    drugs = train_data.rows
    genes = train_data.cols
    sampled_gene_gene_relationships = extract_gene_gene_relationships(drugs, genes,
                                                                      max_neighbors=10) #100 40
    sampled_drug_drug_relationships = extract_gene_gene_relationships(genes, drugs,
                                                                      max_neighbors=30) #40 100

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_loss_r = 0
        epoch_loss_s = 0

        for i, batch in enumerate(train_loader):  #, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True tqdm(train_loader)

            # 从geneids构建同质图进行交互


            drugids, pos = batch
            # 假设 geneids 和 pos 是已给的列表

            neg = [train_neg_samples[drugid.item()][epoch][0] for drugid in drugids]  # 按 epoch 获取负样本
            drugids = drugids.long().cuda(torch.device(device))
            pos = pos.long().cuda(torch.device(device))
            neg = torch.tensor(neg).long().cuda(torch.device(device))
            iids = torch.concat([pos, neg], dim=0)

            optimizer.zero_grad()
            loss, loss_r= model(drugids, iids, pos, neg,  sampled_gene_gene_relationships,sampled_drug_drug_relationships, train_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item()
            epoch_loss_r += loss_r.cpu().item()

        batch_no = len(train_loader)
        epoch_loss = epoch_loss / batch_no
        epoch_loss_r = epoch_loss_r / batch_no
        loss_list.append(epoch_loss)
        loss_r_list.append(epoch_loss_r)
        print(f'Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss} Loss_r: {epoch_loss_r} ')

        # Check if we have a new best loss
        # Check if we have a new best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1  # Save the epoch (1-based)
    torch.save(model.state_dict(), best_model_path)
    print(f'Saved new best model with loss {best_loss} at epoch {best_epoch}')
    return model, best_model_path, best_epoch


def evaluate_model(model, test_pos_samples, test_neg_samples):
    model.eval()
    all_predictions = []
    all_labels = []

    # 计算正样本
    for (i, j) in test_pos_samples:
        drug_id = torch.tensor([i]).long().cuda(torch.device(device))
        gene_id = torch.tensor([j]).long().cuda(torch.device(device))
        pred, _, _ = model(gene_id, drug_id, None, None, None, None, test=True)
        pred = pred.detach().cpu().numpy()  # 使用 detach() 来获取不需要梯度的张量

        all_predictions.append(pred)
        all_labels.append(1)

    # 计算负样本
    for i in test_neg_samples:
        drug_id = torch.tensor([i]).long().cuda(torch.device(device))
        for j in test_neg_samples[i]:
            gene_id = torch.tensor([j]).long().cuda(torch.device(device))
            pred, _, _ = model(gene_id, drug_id, None, None, None,None, test=True)
            pred = pred.detach().cpu().numpy()  # 使用 detach() 来获取不需要梯度的张量

            all_predictions.append(pred)
            all_labels.append(0)

    all_predictions = np.array(all_predictions).ravel()
    all_labels = np.array(all_labels).ravel()

    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    auroc = auc(fpr, tpr)




    print(f"AUROC: {auroc}")


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx]


def main():
    args = parse_args()
    device = 'cuda:' + args.cuda

    train, test = load_data()
    train_coo, test_coo, train_pos_samples, test_pos_samples, train_neg_samples, test_neg_samples = preprocess_data(
        train, test, args.epoch)

    train_labels = [[] for _ in range(train.shape[0])]
    for i in range(len(train_coo.data)):
        row = train_coo.row[i]
        col = train_coo.col[i]
        train_labels[row].append(col)

    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()

    for i in range(len(train_coo.data)):
        train_coo.data[i] = train_coo.data[i] / pow(rowD[train_coo.row[i]] * colD[train_coo.col[i]], 0.5)

    train = train_coo.tocoo()
    train_data = TrnData(train)
    train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    adj_norm = adj_norm.coalesce().cuda(torch.device(device)) #10690, 3227



    train_csr = (train != 0).astype(np.float32)

    model = GCNDDP(adj_norm.shape[0], adj_norm.shape[1], args.d,train_csr,
                    adj_norm, args.gnn_layer, args.temp, args.lambda1, args.lambda2, args.dropout, device)
    model.cuda(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=args.lr)

    model, best_model_path, best_epoch = train_model(train_loader, model, optimizer, train_neg_samples, args.epoch, train_data)

    # Load the best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    print(f'Loaded best model from epoch {best_epoch}')
    evaluate_model(model, test_pos_samples, test_neg_samples)


if __name__ == "__main__":
    main()
