import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.nn.parameter import Parameter
from torch.nn import init

from utils import (vgg_modified,
                   resnet_modified_large,
                   resnet_modified_medium,
                   resnet_modified_small,
                   format_dict, predict_human_readable)

class baseline_attention(nn.Module):
    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    #these seem like decent splits of imsitu, freq = 0,50,100,282 , prediction type can be "max_max" or "max_marginal"
    def __init__(self, encoding, hidden = [32, 32, 32, 32], num_heads = 4, prediction_type = "max_max", ngpus = 1, cnn_type = "resnet_101"):
        super(baseline_attention, self).__init__() 
        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_transform = tv.transforms.Compose([
            tv.transforms.Scale(224),
            tv.transforms.RandomCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])
        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Scale(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.encoding = encoding
        self.hidden = hidden
        self.num_heads = num_heads
        self.prediction_type = prediction_type
        self.n_verbs = encoding.n_verbs()
        #cnn
        print(cnn_type)
        if cnn_type == "resnet_101" : self.cnn = resnet_modified_large()
        elif cnn_type == "resnet_50": self.cnn = resnet_modified_medium()
        elif cnn_type == "resnet_34": self.cnn = resnet_modified_small()
        else: 
            print("unknown base network")
            exit()
        self.rep_size = self.cnn.rep_size()
        
        self.r_n = {r: torch.LongTensor(n).cuda() for r, n in self.encoding.r_n.items()}

        self.pad_v_r = np.array([
            np.pad(v_r, (0, self.encoding.mr-len(v_r)), 'constant', constant_values=(-1, -1))
            for v_r in self.encoding.v_r.values()])
        self.pad_v_r += 1
        self.pad_v_r = torch.LongTensor(self.pad_v_r).cuda()

        # node representaion
        self.linear_r = nn.Linear(self.rep_size, self.encoding.n_roles()*self.hidden[0])

        # attention module
        if len(hidden) > 1:
            self.attention = nn.ModuleList([
                nn.modules.activation.MultiheadAttention(self.hidden[i], self.num_heads, vdim=self.hidden[i+1])
                for i in range(len(hidden)-1)])
        # self.attention = nn.ModuleList([nn.modules.activation.MultiheadAttention(self.rep_size, self.hidden[0])])
        # for i in range(len(hidden)-1):
        #     self.attention += nn.ModuleList(
        #         [nn.modules.activation.MultiheadAttention(self.hidden[i], self.hidden[i+1])])

        # verb potential
        self.linear_v = nn.Linear(self.rep_size, self.encoding.n_verbs())
        # role-noun potential
        # self.linear_rn = nn.ModuleDict({
        #     r: nn.Linear(self.hidden[-1], n) for r, n in self.encoding.r_n.items()
        # })
        # for r, n in self.encoding.r_n.items():
        #     self.linear_rn = nn.Linear(self.hidden[-1], self.encoding.n)
        # noun potentials
        self.linear_n = nn.Linear(self.hidden[-1], self.encoding.n_nouns())

        # print("total encoding vrn : {0}".format(encoding.n_verbrolenoun())) 

    #expects a list of vectors, BxD
    #returns the max index of every vector, max value of each vector and the log_sum_exp of the vector
    def log_sum_exp(self,vec):
        max_score, max_i = vec.max(1)
        max_score_broadcast = max_score.view(-1,1).expand(vec.size())
        return (max_i , max_score,  max_score + (vec - max_score_broadcast).exp().sum(1).log())

    def forward_max(self, images):
        (_,_,_,_,scores, values) = self.forward(images)
        return (scores, values)

    def forward_features(self, images):
        return self.cnn(images)

    def forward(self, image):
        batch_size = image.size()[0]

        rep = self.cnn(image)
        v_potential = self.linear_v(rep) # bsz x n_verbs

        node_rep = self.linear_r(rep).view(
            -1, self.encoding.n_roles(), self.hidden[-1]).transpose(0, 1)
        for attention in self.attention:
            node_rep, _ = attention(node_rep, node_rep, node_rep)

        rn_potential = self.linear_n(node_rep) # roles x bsz x n_nouns
        # rn_potential = list(map(lambda n: rn_potential.index_select(1, n), self.r_n)) # roles x bsz x rn_nouns
        rn_potential_grouped = list(map(
            lambda r, n_potential: n_potential.index_select(1, self.r_n[r]),
            *list(zip(*enumerate(rn_potential)))))

        #first compute the norm
        #step 1 compute the role marginals
        r_maxi_grouped, r_max, rn_marginal = list(zip(*list(map(self.log_sum_exp, rn_potential_grouped)))) # roles x bsz
        r_maxi = list(map(lambda r, r_n: r_n[r_maxi_grouped[r]], *list(zip(*self.r_n.items())))) # roles x bsz
        r_maxi, r_max, rn_marginal = list(map(lambda x: torch.stack(x).transpose(0, 1), [r_maxi, r_max, rn_marginal])) # bsz x roles

        #concat role groups with the padding symbol 
        zeros = torch.zeros(batch_size, 1).cuda() #this is the padding 
        zerosi = torch.LongTensor(batch_size, 1).zero_().cuda()
        rn_marginal = torch.cat([zeros, rn_marginal], 1) # bsz x roles+1
        r_max = torch.cat([zeros, r_max], 1)
        r_maxi = torch.cat([zerosi, r_maxi], 1)     

        #step 2 compute verb marginals
        #we need to reorganize the role potentials so it is BxVxR
        #gather the marginals in the right way
        vrn_marginal_grouped = torch.stack(list(map(lambda r: rn_marginal.index_select(1, r), self.pad_v_r))).transpose(0, 1)
        vr_max_grouped = torch.stack(list(map(lambda r: r_max.index_select(1, r), self.pad_v_r))).transpose(0, 1)
        vr_maxi_grouped = torch.stack(list(map(lambda r: r_maxi.index_select(1, r), self.pad_v_r))).transpose(0, 1)
        # vr_marginal_grouped = r_marginal.index_select(1, v_r).view(batch_size, self.n_verbs, self.encoding.max_roles())
        # vr_max_grouped = r_max.index_select(1, v_r).view(batch_size, self.n_verbs, self.encoding.max_roles())
        # vr_maxi_grouped = r_maxi.index_select(1, v_r).view(batch_size, self.n_verbs, self.encoding.max_roles())

        # product ( sum since we are in log space )
        v_marginal = vrn_marginal_grouped.sum(2).view(batch_size, self.n_verbs) + v_potential

        #step 3 compute the final sum over verbs
        _, _ , norm  = self.log_sum_exp(v_marginal)
        #compute the maxes

        #max_max probs
        v_max = vr_max_grouped.sum(2).view(batch_size, self.n_verbs) + v_potential #these are the scores
        #we don't actually care, we want a max prediction per verb
        #max_max_vi , max_max_v_score = max(v_max,1)
        #max_max_prob = exp(max_max_v_score - norm)
        #max_max_vrn_i = vr_maxi_grouped.gather(1,max_max_vi.view(batch_size,1,1).expand(batch_size,1,self.max_roles))

        #offset so we can use index select... is there a better way to do this?
        #max_marginal probs 
        #max_marg_vi , max_marginal_verb_score = max(v_marginal, 1)
        #max_marginal_prob = exp(max_marginal_verb_score - norm)
        #max_marg_vrn_i = vr_maxi_grouped.gather(1,max_marg_vi.view(batch_size,1,1).expand(batch_size,1,self.max_roles))

        #this potentially does not work with parrelism, in which case we should figure something out 
        if self.prediction_type == "max_max":
            rv = (rep, v_potential, rn_potential, norm, v_max, vr_maxi_grouped) 
        elif self.prediction_type == "max_marginal":
            rv = (rep, v_potential, rn_potential, norm, v_marginal, vr_maxi_grouped) 
        else:
            print("unkown inference type")
            rv = ()
        return rv

  
    #computes log( (1 - exp(x)) * (1 - exp(y)) ) =  1 - exp(y) - exp(x) + exp(y)*exp(x) = 1 - exp(V), so V=  log(exp(y) + exp(x) - exp(x)*exp(y))
    #returns the the log of V 
    def logsumexp_nx_ny_xy(self, x, y):
        #_,_, v = self.log_sum_exp(torch.cat([x, y, torch.log(torch.exp(x+y))]).view(1,3))
        if x > y: 
            return ((y-x).exp() + 1 - y.exp() + 1e-8).log() + x
        else:
            return ((x-y).exp() + 1 - x.exp() + 1e-8).log() + y

    def sum_loss(self, v_potential, vrn_potential, norm, situations, n_refs):
        r"""
        Args:
            v_potential: (batch_size, n_verbs)
            vrn_potential: (batch_size, n_roles, n_nouns)
            norm: (batch_size,)
            situations: (batch_size, 1 + n_refs*max_roles*2)
        """
        #compute the mil losses... perhaps this should be a different method to facilitate parrelism?
        batch_size = v_potential.size()[0]
        mr = self.encoding.max_roles()
        for i in range(0, batch_size):
            _norm = norm[i]
            _v = v_potential[i]
            _vrn = []
            _ref = situations[i]
            for pot in vrn_potential: _vrn.append(pot[i])
            for r in range(0,n_refs):
                v = _ref[0]
                pots = _v[v]
                for (pos,(s, idx, rid)) in enumerate(self.v_roles[v]):
                    pots = pots + _vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]]
                if pots.data[0] > _norm.data[0]: 
                    print("inference error")
                    print(pots)
                    print (_norm)
                if i == 0 and r == 0: loss = pots-_norm
                else: loss = loss + pots - _norm
        return -loss/(batch_size*n_refs)

    def mil_loss(self, v_potential, rn_potential, norm, situations, n_refs): 
        r"""
        Args:
            v_potential: (batch_size, n_verbs)
            rn_potential: (n_roles, batch_size, rn_nouns)
            norm: (batch_size,)
            situations: (batch_size, 1 + n_refs*max_roles*2)
        """
        #compute the mil losses... perhaps this should be a different method to facilitate parrelism?
        batch_size = v_potential.size()[0]
        mr = self.encoding.max_roles()
        for i in range(0, batch_size):
            _norm = norm[i]
            _v = v_potential[i]
            _rn = []
            _ref = situations[i]
            for pot in rn_potential: _rn.append(pot[i])
            for ref in range(0, n_refs):
                v = _ref[0]
                pots = _v[v]
                for (pos, r) in enumerate(self.encoding.v_r[v.item()]):
                    assert _ref[1+2*mr*ref + 2*pos] == r
                    pots = pots + _rn[r][_ref[1 + 2*mr*ref + 2*pos + 1]]
                if pots.item() > _norm.item(): 
                    print("inference error")
                    print(pots)
                    print(_norm)
                if ref == 0: _tot = pots - _norm 
                else : _tot = self.logsumexp_nx_ny_xy(_tot, pots - _norm)
            if i == 0: loss = _tot
            else: loss = loss + _tot
        return -loss/batch_size
