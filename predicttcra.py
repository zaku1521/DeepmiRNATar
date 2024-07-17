import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from optimizer.Radam import *
from emo import EMOLoss

class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, rna_dim=34):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(rna_dim, rna_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    # def gcn(self, input, adj):
    #     # input =[batch,num_node, rna_dim]
    #     # adj = [batch,num_node, num_node]
    #     support = torch.matmul(input, self.weight)
    #     # support =[batch,num_node,rna_dim]
    #     output = torch.bmm(adj, support)
    #     # output = [batch,num_node,rna_dim]
    #     return output

    def make_masks(self, rna_num, gene_num, rna_max_len, gene_max_len):
        N = len(rna_num)  # batch size
        rna_mask = torch.zeros((N, rna_max_len))
        gene_mask = torch.zeros((N, gene_max_len))
        for i in range(N):
            rna_mask[i, :rna_num[i]] = 1   #将掩码矩阵的有效列设置为1，不需要的部分置0
            gene_mask[i, :gene_num[i]] = 1
        rna_mask = rna_mask.unsqueeze(1).unsqueeze(3).cuda()    #将掩码矩阵增加维度，例如原始矩阵是(N, rna_max_len)，在索引为1和3的位置插入一个维度，结果变成四维张量(N,1, rna_max_len,1)
        gene_mask = gene_mask.unsqueeze(1).unsqueeze(2).cuda()    #(N,1, 1,gene_max_len)
        return rna_mask, gene_mask


    def forward(self, rna, gene,rna_num,gene_num):
        # rna = [batch,rna_num, rna_dim]
        # adj = [batch,rna_num, rna_num]
        # gene = [batch,gene len, 100]
        rna_max_len = rna.shape[1]
        gene_max_len = gene.shape[1]
        rna_mask, gene_mask = self.make_masks(rna_num, gene_num, rna_max_len, gene_max_len)
        rna_mask.cuda()
        gene_mask.cuda()
        #rna = self.gcn(rna, adj)
        # rna = torch.unsqueeze(rna, dim=0)
        # rna = [batch size=1 ,rna_num, rna_dim]

        # gene = torch.unsqueeze(gene, dim=0)
        # gene =[ batch size=1,gene len, gene_dim]
        enc_src,rna = self.encoder(gene,rna)
        # enc_src = [batch size, gene len, hid dim]
        ####经过encoder处理  每一个gene的数值维度从100维变成64维
        out = self.decoder(rna, enc_src, rna_mask, gene_mask)
        # out = [batch size, 2]
        # out = torch.squeeze(out, dim=0)
        return out

    def __call__(self, data, train=True):
#???
        rna, gene, correct_interaction, rna_num, gene_num = data
        # rna = rna.to(self.device)
        # adj = adj.to(self.device)
        # gene = gene.to(self.device)
        # correct_interaction = correct_interaction.to(self.device)
        Loss = nn.CrossEntropyLoss()
        # emo_loss = EMOLoss(ignore_index=0, mode=1)



        if train:
            predicted_interaction = self.forward(rna, gene, rna_num, gene_num)
            # cost_embedding = torch.rand(2, 64)
            # loss = emo_loss(predicted_interaction, correct_interaction, cost_embedding)
            loss = Loss(predicted_interaction, correct_interaction)
            #poly1_fc_loss = poly1_focal_loss_torch(torch.tensor(predicted_interaction), torch.tensor(correct_interaction), alpha=0.25, gamma=2,
                                                    #num_classes=2, epsilon=1.0)
            return loss

        else:
            #rna = rna.unsqueeze(0)
            #adj = adj.unsqueeze(0)
            #gene = gene.unsqueeze(0)
            #correct_interaction = correct_interaction.unsqueeze(0)
            predicted_interaction = self.forward(rna, gene, rna_num, gene_num)
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            correct_labels=torch.tensor(correct_labels).cuda()
            predicted_labels=torch.tensor( predicted_labels).cuda()
            predicted_scores=torch.tensor(predicted_scores).cuda()
            return correct_labels, predicted_labels, predicted_scores
