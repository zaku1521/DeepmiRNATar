import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, \
    matthews_corrcoef
from utils import *
from optimizer.Radam import *
from optimizer.lookahead import Lookahead
from gensim.models import Word2Vec
rna2vec = Word2Vec.load("./model/rna.model")
gene2vec = Word2Vec.load("./model/gene.model")


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):

        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        # self.optimizer_inner = Vortex(
        #     [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.batch = batch


    def train(self, dataset,device):
        self.model.train()
        # dataset_iter = iter(dataset)
        # np.random.shuffle(dataset)
        N = len(dataset)
        #print("len of datasets: ", N)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()
        Rnas, Genes, Labels= [], [], []
        # TODO: 进度条
        for idx,val in enumerate(dataset):
            genes,miRNAs,genes_seq,miRNAs_seq,labels = val
            for i in range(len(genes)):
                gene_mutrix = torch.tensor(gene2Matrix(genes_seq[i],gene2vec))  #得到向量矩阵,gene2vec是word2vec模型,[N-k+1,100],N是基因的长度
                miRNA_mutrix = torch.tensor(rna2Matrix(miRNAs_seq[i].replace('\n', ''),rna2vec))
                interaction = torch.tensor(float(labels[i]))
                if torch.cuda.is_available():
                    miRNA_mutrix,gene_mutrix,interaction = miRNA_mutrix.cuda(),gene_mutrix.cuda(),interaction.cuda()
                Rnas.append(miRNA_mutrix)
                Genes.append(gene_mutrix)
                Labels.append(interaction)

            data_pack = pack(Rnas,  Genes, Labels,device)
            loss = self.model(data_pack)
            self.optimizer.zero_grad()
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            rnas, genes, labels = [], [], []
            loss_total += loss.item()
            Rnas, Genes, Labels = [], [], []  #每个列表中有batch个[len,dim]


        return loss_total

