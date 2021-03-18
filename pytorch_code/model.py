#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from WGAT import *


class IEM_layer(nn.Module):   
    def __init__(self, hidden_zise, feat_size):
        super(IEM_layer, self).__init__()
        
        
        self.hidden_zise = hidden_zise
        self.feat_size = feat_size
        self.linear_q = nn.Linear(hidden_zise, feat_size)
        self.linear_k = nn.Linear(hidden_zise, feat_size)

    def forward(self, hidden, mask):
        hidden = hidden * mask.view(mask.shape[0], -1, 1).float()  # batch_size x seq_length x latent_size
        n_batch = hidden.size(0)
        Q = F.sigmoid(self.linear_q(hidden))
        K = F.sigmoid(self.linear_k(hidden))
    
                      
        C = F.sigmoid(torch.matmul(Q, K.transpose(-2,-1)))/ math.sqrt(self.hidden_zise)

        x = trans_to_cuda(torch.eye(hidden.size(1)) - 1)
        xx = x.repeat(n_batch,1,1)
        y = torch.where(xx<0, C, xx)       #b*L*L mask掉对角线的值
        
        T = torch.sum(y,dim=2)
        t = F.softmax(T,dim=1)     #b*L
        
        out = torch.matmul(t.unsqueeze(1),hidden)
            
        return out.squeeze(-2)

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, mask=None):
        key = query
        value = query
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            # scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size     #hidden_size : d
        self.input_size = hidden_size * 2  #input_size  : 2d
        self.gate_size = 3 * hidden_size   #gate_size   : 3d
        
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))  # 3d * 2d
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size)) # 3d * d
        self.b_ih = Parameter(torch.Tensor(self.gate_size))     #3d * 1
        self.b_hh = Parameter(torch.Tensor(self.gate_size))     #3d * 1


        self.linear_in1= nn.Linear(self.gate_size, self.gate_size, bias=True)
        self.linear_in2= nn.Linear(self.gate_size, self.gate_size, bias=True)
        
    #    self.weiged_gat1=  GraphAttentionLayer(self.hidden_size, self.hidden_size)
     #   self.weiged_gat2=  GraphAttentionLayer(self.hidden_size, self.hidden_size)
        
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))  #d * 1
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))  #d * 1
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
        

    def GNNCell(self, A, hidden):
        
  #      input_in = self.weiged_gat1(hidden,A[:, :, :A.shape[1]])
#        inputs_in = self.linear_edge_in(inputs_in)
   #     input_out = self.weiged_gat2(hidden,A[:, :, A.shape[1]: 2 * A.shape[1]])
#        inputs_out = self.linear_edge_out(inputs_out)

        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)

      
        test = torch.cat([inputs,hidden], 2)
        test1 = self.linear_in1(test)
        test2 = self.linear_in2(test)


        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        
        gi = gi+test1
        gh = gh+test2
        
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)

        
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_out = nn.Linear(2*self.hidden_size,self.hidden_size, bias=True)
        


        self.iem = IEM_layer(self.hidden_size,self.hidden_size)
       # self.multihead_attn = MultiHeadedAttention(4, self.hidden_size, 0.2).cuda() 
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):  #mask用来过滤之前用0来填缺的商品
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        
   #     attn_output = hidden

    #    attn_output = self.multihead_attn(attn_output)
        
    #    a = attn_output[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # use last one as global interest

        IEM_output = self.iem(hidden,mask)
        hn = IEM_output  # use last one as global interest

        x = torch.cat([ht,hn],1)
        y = self.linear_out(x)

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(y, b.transpose(1, 0))
        return scores
    
    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden



def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    
    hidden = model(items, A)
    # print("hidden:\n")
    # print(hidden.size(),"\n")
    
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    # print("seq_hidden:\n")
    # print(seq_hidden.size(),"\n")
    
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    
    for i, j in zip(slices, np.arange(len(slices))):
        # print("i:\n")
        # print(i)
        # print("\nj:\n")
        # print(j,"\n")
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
