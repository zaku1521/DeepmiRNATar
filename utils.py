from functools import wraps
import torch
from tqdm import tqdm as _tqdm
import numpy as np

@wraps(_tqdm)
def tqdm(*args, **kwargs):
    with _tqdm(*args, **kwargs) as t:
        try:
            for _ in t:
                yield _
        except KeyboardInterrupt:
            t.close()
            raise KeyboardInterrupt

def seq_to_kmers(seq, k=4):   #生成长度为K，滑动大小为1的序列
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
        k (int), default 3
    Returns:
        List containing a list of kmers.
    """
    #seq=seq.astype(object)

    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]


def get_gene_embedding(model, gene):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((len(gene), 100))   #存储基因的嵌入表示[基因长度,100]
    i = 0
    for word in gene:
        vec[i, ] = model.wv[word]   #将每个长度为4的序列转换成100维的向量，即每一行有100个数据
        i += 1
    return vec

def gene2Matrix(geneSeq,gene2vec):
    return get_gene_embedding(gene2vec, seq_to_kmers(geneSeq))

def get_rna_embedding(model, rna):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((len(rna), 34))
    i = 0
    for word in rna:
        vec[i, ] = model.wv[word]
        i += 1
    return vec

def rna2Matrix(miRNASeq,rna2vec):
    return  get_rna_embedding(rna2vec,seq_to_kmers(miRNASeq))

def get_all_embedding(model, all):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((len(all), 34))
    i = 0
    for word in all:
        vec[i, ] = model.wv[word]
        i += 1
    return vec

def all2Matrix(miRNASeq,rna2vec):
    return  get_all_embedding(rna2vec,seq_to_kmers(miRNASeq))
# 获取每次输入的基因，rna,合并的数据
def pack(rnas, genes, labels,  device):
    #print("device: ", device)
    rnas_len = 0
    genes_len = 0
    N = len(rnas)
    rna_num = []    #每次batch的每个rna的长度(k个字符算一个长度）
    for rna in rnas:
        rna_num.append(rna.shape[0])  #shape[0]表示第一个维度的大小，对rna来说，这个维度通常代表序列的长度   rna是[len,34]，rnas是[batch,len,34]
        if rna.shape[0] >= rnas_len:
            rnas_len = rna.shape[0]   #rnas_len是每个batch中最长的rna长度值
    gene_num = []
    for gene in genes:
        gene_num.append(gene.shape[0])
        if gene.shape[0] >= genes_len:
            genes_len = gene.shape[0]
    rnas_new = torch.zeros((N, rnas_len, 34)).cuda() #N是rna的数量即batch，rnas_len最长的rna长度值
    i = 0
    for rna in rnas:
        a_len = rna.shape[0]
        rnas_new[i, :a_len, :] = rna    #a_len<=rnas_len,多余部分为0
        i += 1
    genes_new = torch.zeros((N, genes_len, 100)).cuda()
    i = 0
    for gene in genes:
        a_len = gene.shape[0]
        genes_new[i, :a_len, :] = gene
        i += 1
    labels_new = torch.zeros(N, dtype=torch.long).cuda()
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    rna_num =torch.tensor(rna_num)
    gene_num = torch.tensor(gene_num)
    return (rnas_new, genes_new, labels_new, rna_num, gene_num)


# def testpack(rnas, genes,  device):
#     print("device: ", device)
#     rnas_len = 0
#     genes_len = 0
#     N = len(rnas)
#     rna_num = []
#     for rna in rnas:
#         rna_num.append(rna.shape[0])
#         if rna.shape[0] >= rnas_len:
#             rnas_len = rna.shape[0]
#     gene_num = []
#     for gene in genes:
#         gene_num.append(gene.shape[0])
#         if gene.shape[0] >= genes_len:
#             genes_len = gene.shape[0]
#     rnas_new = torch.zeros((N, rnas_len, 34), device=device)
#     i = 0
#     for rna in rnas:
#         a_len = rna.shape[0]
#         rnas_new[i, :a_len, :] = rna
#         i += 1
#     # adjs_new = torch.zeros((N, rnas_len, rnas_len), device=device)
#     # i = 0
#     # for adj in adjs:
#     #     a_len = adj.shape[0]
#     #     #FIXME cuda()
#     #     adj = adj.cuda()
#     #     adj = adj + torch.eye(a_len, device=device)
#     #     adjs_new[i, :a_len, :a_len] = adj
#     #     i += 1
#     genes_new = torch.zeros((N, genes_len, 100), device=device)
#     i = 0
#     for gene in genes:
#         a_len = gene.shape[0]
#         genes_new[i, :a_len, :] = gene
#         i += 1
#     labels_new = torch.zeros(N, dtype=torch.long, device=device)
#     i = 0
#     for label in labels:
#         labels_new[i] = label
#         i += 1
#     return (rnas_new, genes_new, labels_new, rna_num, gene_num)

# def pack(seqs, labels,  device):
#     #print("device: ", device)
#     seqs_len = 0
#     N = len(seqs)
#     seqs_num = []    #每次batch的每个rna的长度(k个字符算一个长度）
#     for seq in seqs:
#         seqs_num.append(seq.shape[0])  #shape[0]表示第一个维度的大小，对rna来说，这个维度通常代表序列的长度   rna是[len,34]，rnas是[batch,len,34]
#         if seq.shape[0] >= seqs_len:
#             seqs_len = seq.shape[0]   #rnas_len是每个batch中最长的rna长度值
#     seqs_new = torch.zeros((N, seqs_len, 34)).cuda() #N是rna的数量即batch，rnas_len最长的rna长度值
#     i = 0
#     for seq in seqs:
#         a_len = seq.shape[0]
#         seqs_new[i, :a_len, :] = seq   #a_len<=rnas_len,多余部分为0
#         i += 1
#     labels_new = torch.zeros(N, dtype=torch.long).cuda()
#     i = 0
#     for label in labels:
#         labels_new[i] = label
#         i += 1
#     seqs_num =torch.tensor(seqs_num)
#     return (seqs_new, labels_new, seqs_num)
#
