import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve
import random


def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label) > 0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit += 1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc + 2))
            all_recall = all_recall + hit / len(label)
            all_ndcg = all_ndcg + dcg / idcg
            user_num += 1
    return all_recall / user_num, all_ndcg / user_num


def mm_auc(uids, pred_score, test_labels, train_labels, y_pred, y_true):
    for i in range(len(uids)):
        uid = uids[i]
        item_scores = pred_score[i]
        pos = test_labels[uid]
        train_item_ass = train_labels[uid]
        train_test_ass = pos + train_item_ass
        diff = list(set(range(len(item_scores))) - set(train_test_ass))
        random.shuffle(diff)
        neg = diff[0:len(pos)]
        y_true += [1] * len(pos)
        y_true += [0] * len(pos)
        for item in pos:
            y_pred.append(item_scores[item])
        for item in neg:
            y_pred.append(item_scores[item])
    return y_pred, y_true


def evalRanking_valid(self, dataset):
    predict_list = []
    actual_list = []
    # path1 = "./dataset/lncRNA_drug_not_Mutation_p-value_0.05/ass.txt"
    path1 = "./dataset/drug_disease/ass.txt"
    ass = {}

    with open(path1, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            line[0], line[1] = int(line[0]), int(line[1])
            # if line[0] == 0 or line[1] == 0:
            # print(f'0!!!!!!')
            if line[0] not in ass.keys():
                ass[line[0]] = []
            ass[line[0]].append(int(line[1]))

    negative_dict = {}
    for key in dataset:
        # if key not in self.user
        lg = len(dataset[key])
        all_i = set(ass[key])
        cu_i = set([i for i in range(self.data.item_num)])
        need_i = list(cu_i - all_i)
        random.shuffle(need_i)
        need_i = need_i[:lg]
        negative_dict[key] = need_i
    # print(negative_dict)

    # 保存分数用的
    miRNA = []
    drug = []

    # for user in ass2.keys():
    for _, user in enumerate(dataset):
        itemSet = {}
        # print(f'user: {user}')
        predictedItems = self.predict(user)
        rated_list, li = self.data.user_rated(user)
        for item in rated_list:
            predictedItems[self.data.item[item]] = -10e8
        # if predictedItems == pd.NA:
        #     continue
        # print(f'predictedItems: {predictedItems}')
        # result_score.append(predictedItems)
        for id, rating in enumerate(predictedItems):
            # print(f'id: {id}; rating: {rating}')
            itemSet[self.data.id2item[id]] = rating

        for i in dataset[user]:
            miRNA.append(user)
            drug.append(i)
            if i in itemSet:
                va = itemSet[i]
                actual_list.append(1)
                predict_list.append(va)
            else:
                actual_list.append(1)
                # predict_list.append(sum(itemSet.values())/len(itemSet.keys()))
                predict_list.append(1)

        for i in negative_dict[user]:
            miRNA.append(user)
            drug.append(i)
            # i = str(i)
            if i in itemSet:
                va = itemSet[i]
                actual_list.append(0)
                predict_list.append(va)
            else:
                actual_list.append(0)
                # predict_list.append(sum(itemSet.values()) / len(itemSet.keys()))
                predict_list.append(0)

    # 评价指标
    # print(f'actual_list: {actual_list}\n'
    #       f'predict_list: {predict_list}')
    fpr, tpr, _ = roc_curve(actual_list, predict_list)
    # np.save("fpr.npy", fpr)
    # np.save("tpr.npy", tpr)
    auroc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(actual_list, predict_list)
    # np.save("precision.npy", precision)
    # np.save("recall.npy", recall)
    aupr = auc(recall, precision)
    if dataset == self.data.test_set:
        print(f"test_set: auc: {auroc}; aupr: {aupr}")
    else:
        print(f'valid_set: auc: {auroc}; aupr: {aupr}')
    return auroc, aupr, fpr, tpr, precision, recall

def GIP_kernel (Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    #initate a matrix as result matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    #calculate the result matrix
    for i in range(nc):
        for j in range(nc):
            #calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i,:] - Asso_RNA_Dis[j,:]))
            if r == 0:
                matrix[i][j]=0
            elif i==j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e**(-temp_up/r)
    return matrix

def getGosiR (Asso_RNA_Dis):
# calculate the r in GOsi Kerel
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i,:])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r

def get_syn_sim (A, k1, k2):
    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)
    for i in range(k1):
        GIP_m_sim[i,i] = 1
    for j in range(k2):
        GIP_d_sim[j,j] = 1
    Pm_final = GIP_m_sim
    Pd_final = GIP_d_sim
    return Pm_final, Pd_final

def get_lapl_matrix(sim):
    m,n = sim.shape
    lap_matrix_tep = np.zeros([m,m])
    for i in range(m):
        lap_matrix_tep[i,i] = np.sum(sim[i,:])
    lap_matrix = lap_matrix_tep - sim
    return lap_matrix, lap_matrix_tep


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)


def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs = emb[cols] * torch.unsqueeze(sp.values(), dim=1)
    result = torch.zeros((sp.shape[0], emb.shape[1])).cuda(torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]
