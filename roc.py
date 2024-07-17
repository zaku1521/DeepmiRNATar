import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, \
    matthews_corrcoef
from transformers import AutoModel, AutoTokenizer
from utils import *
from train import *
from optimizer.Radam import *
from optimizer.lookahead import Lookahead
from gensim.models import Word2Vec
from sklearn.metrics import roc_curve, auc
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# model_name = "./model/output4-2epoch"
# genemodel = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        # print("len of test: ", N)
        T, Y, S = [], [], []
        with torch.no_grad():
            # for idx,val in tqdm(enumerate(dataset.dataset), ascii=True):
            #     Rnas, Genes, Labels = [], [], []
            #     genes, genes_seq, miRNAs, miRNAs_seq, labels = val
            for idx,val in enumerate(dataset):
                Rnas, Genes, Labels = [], [], []
                genes, miRNAs, genes_seq, miRNAs_seq, labels = val
                gene_mutrix = torch.tensor(gene2Matrix(genes_seq[0], gene2vec))
                # k_mers = seq_to_kmers(genes_seq[0], k=4)
                # encoded_k_mers = [genemodel(**tokenizer(k_mer, return_tensors="pt"))['last_hidden_state'][:, 0, :] for
                #                   k_mer in k_mers]
                # gene_mutrix = torch.cat(encoded_k_mers, dim=0)
                miRNA_mutrix = torch.tensor(rna2Matrix(miRNAs_seq[0].replace('\n', ''), rna2vec))
                interaction = torch.tensor(float(labels))
                if torch.cuda.is_available():
                    miRNA_mutrix, gene_mutrix, interaction = miRNA_mutrix.cuda(), gene_mutrix.cuda(), interaction.cuda()
                Rnas.append(miRNA_mutrix)
                Genes.append(gene_mutrix)
                Labels.append(interaction)
                data = pack(Rnas, Genes, Labels, self.model.device_ids)
                correct_labels, predicted_labels, predicted_scores = self.model(data=data, train=False)
                T.extend(correct_labels.detach().cpu().numpy())
                Y.extend(predicted_labels.detach().cpu().numpy())
                S.extend(predicted_scores.detach().cpu().numpy())

        C = confusion_matrix(T, Y)
        acc = accuracy_score(T, Y)
        pre = precision_score(T, Y, average='binary')
        rec = recall_score(T, Y, average='binary')
        p, r, _ = precision_recall_curve(T, S)
        f1 = f1_score(T, Y)
        AUC = roc_auc_score(T, S)
        PRC = auc(r, p)
        sen = rec
        spe = C[1][1]*1.0/(C[0][1]+C[1][1])
        mcc = matthews_corrcoef(T, Y,sample_weight=None)

        fpr, tpr, thersholds = roc_curve(T, S)

        ##测试集时画曲线，验证集不画
        ##画ROC曲线
        # ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.rcParams.update({'font.size': 18})
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Fix for Chinese characters
        plt.rcParams['axes.unicode_minus'] = False  # Fix for minus sign
        ax.plot(fpr, tpr, linewidth=2, color='black',label='DeepmiRNATar(AUC=0.9915)')
        ax.plot([0, 1], [0, 1], color=(0.6,0.6,0.6), lw=2, linestyle='--', label='Reference (AUC=0.5)')
        ax.set_xlabel('False Positive Rate', size=16)
        ax.set_ylabel('True Positive Rate', size=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title('ROC of the test dataset', fontsize=16)
        ax.legend(loc='lower right')

        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig('roc.png', dpi=350, bbox_inches='tight')
        plt.show()

        # PRC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(r, p, color='black', lw=2, label=f'DeepmiRNATar(PRC={PRC:.4f})')
        ax.set_xlabel('Recall', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
        # ax.set_ylim([0.0, 1.05])
        # ax.set_xlim([0.0, 1.05])
        ax.set_title('PRC of the test dataset', fontsize=16)
        ax.legend(loc="lower left")
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig('prc.png', dpi=350, bbox_inches='tight')
        plt.show()
        return acc,pre,rec,f1,AUC,PRC,sen,spe,mcc

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)