# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/17 8:36
@author: LiFan Chen
@Filename: model.py
@Software: PyCharm
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, \
    matthews_corrcoef
from optimizer.Radam import *
from optimizer.lookahead import Lookahead
from gensim.models import Word2Vec
from sklearn.metrics import roc_curve, auc


class MultiHeadAttentionWithACmix(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device, acmix_dim=256):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.acmix_dim = acmix_dim

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        # 初始化随机的 acmix 张量
        self.register_buffer('acmix', torch.randn(1, acmix_dim, 1))
        # ACmix parameters
        self.acmix_linear = nn.Linear(acmix_dim, hid_dim)
        self.acmix_conv = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1)

        self.dp = nn.Dropout(dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
        self.gamma = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = F.softmax(energy, dim=-1)
        attention = self.dp(attention)

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # 应用 ACmix
        self.acmix = self.acmix.transpose(1, 2)
        acmix_linear_output = self.acmix_linear(self.acmix)  # 将 acmix 线性映射到 hid_dim
        self.acmix = self.acmix.transpose(1, 2)
        acmix_linear_output = acmix_linear_output.transpose(1, 2)  # 调整维度
        # 进行卷积
        acmix_conv_output = self.acmix_conv(acmix_linear_output)
        acmix_conv_output = acmix_conv_output.transpose(1, 2)  # 调整维度

        x = self.gamma * x + acmix_conv_output + query  # 将 ACmix 的输出加到原始多头注意力输出上

        x = self.fc(x)

        return x

class TextSPP(nn.Module):
    def __init__(self, SPPSize, name='textSpp'):
        super(TextSPP, self).__init__()
        self.name = name
        self.spp = nn.AdaptiveAvgPool1d(SPPSize)
    def forward(self, x):
        return self.spp(x.cpu()).to(x.device)


class TextCNN(nn.Module):
    def __init__(self, feaSize, contextSizeList, filterNum, name='textCNN'):
        super(TextCNN, self).__init__()
        self.name = name
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=feaSize, out_channels=filterNum, kernel_size=contextSizeList[i],
                              padding=contextSizeList[i] // 2),
                    nn.ReLU()
                    # nn.AdaptiveMaxPool1d(1)
                )
            )
        self.conv1dList = nn.ModuleList(moduleList)

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.permute(0, 2, 1)
        # print(x.size())
        x = torch.cat([conv(x) for conv in self.conv1dList], dim=1)

        return x  # => batchSize × scaleNum*filterNum × seq_len
class TextBiLSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, name='textBiLSTM'):
        super(TextBiLSTM, self).__init__()
        self.name = name
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=True, batch_first=True)
    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biLSTM(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]
        return output # output: batchSize × seqLen × hiddenSize*2

class TextBiGRU(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, name='textBiGRU'):
        super(TextBiGRU, self).__init__()
        self.name = name
        self.biGRU = nn.GRU(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biGRU(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]
        return output # output: batchSize × seqLen × hiddenSize*2

# 这个其实是用的TextCNN+BiGRU或者BiLSTM去处理两个序列
class Encoder(nn.Module):
    """gene feature extraction."""
    def __init__(self, gene_dim,rna_dim, hid_dim, dropout, device,SPPSize, feaSize, filterNum, contextSizeList):
        super().__init__()

        # assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = gene_dim
        self.hid_dim = hid_dim
        self.feaSize = feaSize
        self.dropout = dropout
        self.filterNum = filterNum
        self.device = device
        self.contextSizeList=contextSizeList
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).cuda()
        self.dropout = nn.Dropout(dropout)
        self.fc1=nn.Linear(len(self.contextSizeList)*self.filterNum,self.hid_dim)
        self.ft = nn.Linear(rna_dim, hid_dim)
        self.fc = nn.Linear(self.input_dim, self.hid_dim) ##全连接层，用于实现降维或者分类，全连接层后常用softmax去实现预测结果转变为概率
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim) ##channal方向做归一化
        self.textSPP = TextSPP(SPPSize).to(device)
        self.textCNN = TextCNN(feaSize, contextSizeList, filterNum).to(device)
        self.bilstm=TextBiLSTM(feaSize,hid_dim)
        self.bigru = TextBiGRU(feaSize, hid_dim)

    def forward(self, gene,rna):
        gene=self.dropout(gene)
        gene=self.textSPP(gene)
        gene=self.textCNN(gene)
        gene=gene.permute(0,2,1)
        gene= self.fc1(gene)  # Final linear layer
        gene=self.bilstm(gene)
        gene=F.glu(gene,dim=2)
        gene = self.ln(gene)

        rna = self.dropout(rna)
        rna = self.textSPP(rna)
        rna = self.textCNN(rna)
        rna = rna.permute(0, 2, 1)
        rna = self.fc1(rna)  # Final linear layer
        rna = self.bilstm(rna)
        rna = F.glu(rna, dim=2)
        rna = self.ln(rna)
        return gene,rna



class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim,dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout) ##Dropout是一个正则化网络，防止过拟合，dropout是丢失率，在深层网络下，0.5的丢失率是较好的选择，但是在浅层的网络下，丢失率应该低于0.2

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class Decoder(nn.Module):
    """ rna feature extraction."""
    def __init__(self, rna_dim, hid_dim, n_layers, n_heads, pf_dim, multihead_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = rna_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.self_attention = multihead_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = multihead_attention(hid_dim, n_heads, dropout, device)
        self.ft = nn.Linear(rna_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.gn = nn.GroupNorm(8, 256)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, src, src, src_mask)))

        # trg_=trg
        trg = self.ln(trg + self.do(self.pf(trg)))

        # trg = [batch size, rna len, hid dim]
        """Use norm to determine which rna is significant. """
        #???
        norm = torch.norm(trg, dim=2)
        # norm = [batch size,rna len]
        norm = F.softmax(norm, dim=1)
        sum = torch.zeros((trg.shape[0], self.hid_dim)).cuda()

        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j, ]
                v = v * norm[i, j]
                sum[i, ] += v
        # sum = [batch size,hid_dim]
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)


        return label

