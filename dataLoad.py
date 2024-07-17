import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import numpy as np


class init_dataset(Dataset):
    def __init__(self,data_address):
        self.mydata = pd.read_csv(data_address)
        self.data = self.mydata[['gene','miRNA','gene_sequence','miRNA_sequence']]
        self.label = self.mydata[['interaction']]
        self.length = self.mydata.shape[0]
    def __getitem__(self, idx):
        label = np.array(self.label.iloc[idx]).tolist()
        data = np.array(self.data.iloc[idx]).tolist()

        return data+label
    def __len__(self):
        return self.length

    # def get_data(self, slice):    #slice是数据的序号
    #     # d = [self.mydata.iloc[i] for i in slice]     #d=[a,b,c的shape[1，954]]
    #     # return d         #输出的已经是向量,slice个数据，每个数据是列表，有三个元素
    #     return [self.mydata.iloc[i] for i in slice]

    # def get_data(self, slice):
    #     data_list = []
    #     for i in slice:
    #         gene_seq = [int(char) for char in self.mydata['gene_sequence'].iloc[i]]  # 转换为数字列表
    #         miRNA_seq = [int(char) for char in self.mydata['miRNA_sequence'].iloc[i]]  # 转换为数字列表
    #         gene = np.array(self.mydata['gene'].iloc[i])
    #         miRNA = np.array(self.mydata['miRNA'].iloc[i])
    #         label = np.array(self.mydata['interaction'].iloc[i])
    #
    #         data_list.append(
    #             {'gene': gene, 'miRNA': miRNA, 'gene_sequence': gene_seq, 'miRNA_sequence': miRNA_seq,
    #              'interaction': label})
    #
    #     return data_list

# def collate_fn(batch):
#     # 在这里，batch 是一个列表，其中每个元素是一个包含原始数据的 DataFrame 行
#     # 可以根据需要进行处理，将这些行整理成一个批次的格式
#     # 例如，将每列的数据整理为张量
#
#     # 假设数据结构为：[gene, miRNA, gene_sequence, miRNA_sequence, interaction]
#     genes_list, miRNAs_list, gene_seqs_list, miRNA_seqs_list, labels_list = [], [], [], [], []
#
#     for item in batch:
#         genes_list.append(item['gene'])
#         miRNAs_list.append(item['miRNA'])
#         gene_seqs_list.append(item['gene_sequence'])
#         miRNA_seqs_list.append(item['miRNA_sequence'])
#         labels_list.append(item['interaction'])
#
#     # 将列表中的数据整理为张量
#     genes_tensor = torch.stack([torch.from_numpy(seq) for seq in genes_list])
#     miRNAs_tensor = torch.stack([torch.from_numpy(seq) for seq in miRNAs_list])
#     gene_seqs_tensor = torch.stack([torch.from_numpy(seq) for seq in gene_seqs_list])
#     miRNA_seqs_tensor = torch.stack([torch.from_numpy(seq) for seq in miRNA_seqs_list])
#     labels_tensor = torch.tensor(labels_list)
#
#     # 返回整理好的批次数据
#     return {'gene': genes_tensor, 'miRNA': miRNAs_tensor, 'gene_sequence': gene_seqs_tensor,
#             'miRNA_sequence': miRNA_seqs_tensor, 'interaction': labels_tensor}
# def collate_fn(batch):
#     return [init_dataset.__getitem__(item) for item in batch]
        # Rnas, Genes, Labels= [], [], []
        # for d in data:
        #     genes,miRNAs,genes_seq,miRNAs_seq,labels = d[0], d[1], d[3], d[4],d[2]
        #
        #     Genes.append(genes_seq)
        #     Rnas.append(miRNAs_seq)
        #     Labels.append(labels)
        #
        #
        # return Batch.from_data_list(d1_list), Batch.from_data_list(d2_list), torch.tensor(label_list)

    # def __init__(self, data_address, fold_index):
    #     super(init_dataset, self).__init__(data_address)
    #     self.fold_index = fold_index

    # def get_data(self, slice):
    #     data_list = []
    #     for i in slice:
    #         gene_seq = self.mydata['gene_sequence'].iloc[i]
    #         miRNA_seq = self.mydata['miRNA_sequence'].iloc[i]
    #         gene = self.mydata['gene'].iloc[i]
    #         miRNA = self.mydata['miRNA'].iloc[i]
    #         label = self.mydata['interaction'].iloc[i]
    #
    #         data_list.append(
    #             {'gene': gene, 'miRNA': miRNA, 'gene_sequence': gene_seq, 'miRNA_sequence': miRNA_seq,
    #              'interaction': label})
    #
    #     return data_list


# def collate_fn(batch):
#     # 在这里，batch 是一个列表，其中每个元素是一个包含原始数据的 DataFrame 行
#     # 可以根据需要进行处理，将这些行整理成一个批次的格式
#     # 例如，将每列的数据整理为张量
#
#     # 假设数据结构为 [data, label]
#     data_list, label_list = [], []
#
#     for item in batch:
#         data_list.append(item['data'])
#         label_list.append(item['label'])
#
#     # 这里你可以根据模型的输入格式对数据进行处理
#     # 以下是一个简单的例子，假设 data 是一个包含两个字段的列表
#     # 你需要根据实际情况进行修改
#     processed_data = torch.stack([torch.tensor(data) for data in data_list])
#     labels_tensor = torch.tensor(label_list)
#
#     # 返回整理好的批次数据
#     return {'data': processed_data, 'label': labels_tensor}

