import argparse
import json
import os
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

from imsitu import imSituSituation
from baseline_crf import vgg_modified as VGGModified
import wandb
# , predict_n_loss="mil"
#         self.predict_n_loss = predict_n_loss
#             if predict_n_loss == "mil":
#                 self.loss = self.mil_loss
#             elif predict_n_loss == "sum":
#                 self.loss = self.sum_loss
#             else:
#                 assert False, "Wrong predict_n_loss"

class imSituVerbRoleNounUNKEncoder:

    def n_verbs(self): return len(self.v_id)
    def n_nouns(self): return len(self.n_id)
    def n_roles(self): return len(self.r_id)
    def n_frames(self): return len(self.fr_v)
    def verbposition_role(self, v, i): return self.v_r[v][i]
    def verb_nroles(self, v): return len(self.v_r[v])
    def max_roles(self): return self.mr
    def pad_symbol(self): return -1
    def unk_symbol(self): return -2

    def __init__(self, dataset, top_n_noun=2000):
        self.v_id = {}
        self.id_v = {}

        self.r_id = {}
        self.id_r = {}

        self.fr_id = {}
        self.id_fr = {}

        self.id_n = {0: "UNK"}
        self.n_id = {"UNK": 0}
        self.n_count = {}

        self.mr = 0

        self.v_r = {}
        self.fr_v = {}

        for (image, annotation) in dataset.items():
            v = annotation["verb"]
            if v not in self.v_id:
                _id = len(self.v_id)
                self.v_id[v] = _id
                self.id_v[_id] = v
                self.v_r[_id] = []
            vid = self.v_id[v]
            for frame in annotation["frames"]:
                for (r, n) in frame.items():
                    if r not in self.r_id:
                        _id = len(self.r_id)
                        self.r_id[r] = _id
                        self.id_r[_id] = r

                    if n not in self.n_count:
                        self.n_count[n] = 1
                    else:
                        self.n_count[n] += 1

                    rid = self.r_id[r]
                    if rid not in self.v_r[vid]:
                        self.v_r[vid].append(rid)
                fr = frozenset([self.r_id[r] for r in frame])
                if fr not in self.fr_id:
                    fid = len(self.fr_id)
                    self.fr_id[fr] = fid
                    self.id_fr[fid] = fr
                if fr not in self.fr_v:
                    self.fr_v[fr] = []
                if vid not in self.fr_v[fr]:
                    self.fr_v[fr].append(vid)

        sorted_n_count = sorted(self.n_count.items(), key=lambda x: x[1], reverse=True)
        for n, _ in sorted_n_count[:top_n_noun]:
            _id = len(self.n_id)
            self.n_id[n] = _id
            self.id_n[_id] = n

        for (v, rs) in self.v_r.items():
            if len(rs) > self.mr:
                self.mr = len(rs)

        for (v, vid) in self.v_id.items():
            self.v_r[vid] = sorted(self.v_r[vid])
        
    def encode(self, situation):
        rv = {}
        verb = self.v_id[situation["verb"]]
        rv["verb"] = verb
        rv["frames"] = []
        for frame in situation["frames"]:
            _e = []
            for (r, n) in frame.items():
                _rid = self.r_id[r] if r in self.r_id else self.unk_symbol()
                _nid = self.n_id[n] if n in self.n_id else self.n_id["UNK"]
                _e.append((_rid, _nid))
            rv["frames"].append(_e)
        return rv

    def decode(self, situation):
        verb = self.id_v[situation["verb"]]
        rv = {"verb": verb, "frames": []}
        for frame in situation["frames"]:
            _fr = {}
            for (r, n) in frame.items():
                _fr[self.id_r[r]] = self.id_n[n]
            rv["frames"].append(_fr)
        return rv

    # takes a list of situations
    def to_tensor(self, situations, use_role=True, use_verb=True):
        rv = []
        for situation in situations:
            _rv = self.encode(situation)
            verb = _rv["verb"]
            items = []
            if use_verb:
                items.append(verb)
            for frame in _rv["frames"]:
                # sort roles
                _f = sorted(frame, key=lambda x: x[0])
                k = 0
                for (r, n) in _f:
                    if use_role:
                        items.append(r)
                    items.append(n)
                    k += 1
                while k < self.mr:
                    if use_role:
                        items.append(self.pad_symbol())
                    items.append(self.pad_symbol())
                    k += 1
            rv.append(torch.LongTensor(items))
        return torch.cat(rv)

    # the tensor is BATCH x VERB X FRAME
    def to_situation(self, tensor):
        (batch, verbd, _) = tensor.size()
        rv = []
        for b in range(0, batch):
            _tensor = tensor[b]
            #_rv = []
            for verb in range(0, verbd):
                args = []
                __tensor = _tensor[verb]
                for j in range(0, self.verb_nroles(verb)):
                    n = __tensor.data[j]
                    args.append((self.verbposition_role(verb, j), n))
                situation = {"verb": verb, "frames": [args]}
                rv.append(self.decode(situation))
            # rv.append(_rv)
        return rv


class BackBone(nn.Module):
    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    def __init__(self, encoding, cnn_type):
        super(BackBone, self).__init__()
        self.encoding = encoding

        self.criteria = nn.CrossEntropyLoss(ignore_index=-1)
        self.criteria_ = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

        if cnn_type == "vgg":
            self.cnn = VGGModified()
        self.rep_size = self.cnn.rep_size()

        self.linear_v = nn.Linear(self.rep_size, self.encoding.n_verbs())
        self.linear_n = nn.Linear(self.rep_size, self.encoding.n_nouns())
        self.linear_r = nn.Linear(self.rep_size, self.encoding.n_roles())
        self.linear_f = nn.Linear(self.rep_size, self.encoding.n_frames())

        self.normalize = tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.RandomCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])                

    def forward(self, image, predict):
        rep = self.cnn(image)

        res = {}
        if "verb" in predict:
            res["verb"] = self.linear_v(rep)
        if "noun" in predict:
            res["noun"] = self.linear_n(rep)    
        if "role" in predict:
            res["role"] = self.linear_r(rep)    
        if "frame" in predict:
            res["frame"] = self.linear_f(rep)    
        return res
    
def train_batch(model, optimizer, input_, target, args):
    res = {}
    optimizer.zero_grad()
    pred = model(input_, args.predict)
    if "verb" == args.predict:
        loss = model.criteria(pred["verb"], target[:, 0])
        loss.backward()
        optimizer.step()
        res['verb_loss'] = loss.detach().item()
    if "role" == args.predict:
        loss = model.criteria(pred["role"].repeat(18, 1), target[:, 1::2].transpose(0, 1).flatten())
        loss.backward()
        optimizer.step()
        res['role_loss'] = loss.detach().item()
    if "frame" == args.predict:
        fr_target = torch.tensor([
            model.encoding.fr_id[frozenset([fr for fr in trg if fr >= 0])]
            for trg in target[:, 1::2].cpu().tolist()]).to(args.gpu)
        loss = model.criteria(pred["frame"], fr_target)
        loss.backward()
        optimizer.step()
        res['frame_loss'] = loss.detach().item()
    if "noun" == args.predict:
        loss = model.criteria(pred["noun"].repeat(18, 1), target[:, 2::2].transpose(0, 1).flatten())
        loss.backward()
        optimizer.step()
        res['noun_loss'] = loss.detach().item()
    return res

def evaluation(model, eval_loader, args):
    with torch.no_grad():
        eval_loop = tqdm(eval_loader, total=len(eval_loader))
        # eval_loop = eval_loader
        columns = ["Target vid", "Target verb", "Top 10 Pred vid", "Top 10 Pred verb"]
        pd_res = pd.DataFrame(columns=columns)
        res = {"total": 0, 
               "verb_loss": .0, "verb_correct": 0,
               "noun_loss": .0, "noun_correct": 0, #"noun_all_correct": 0,
               "role_loss": .0, "role_correct": 0, #"role_all_correct": 0,
               "frame_loss": .0, "frame_correct": 0}
        for idx, img, target in eval_loop:
            batch_size = img.shape[0]
            input_ = img.to(args.gpu)
            target = target.to(args.gpu)

            pred = model(input_, args.predict)
            res["total"] += batch_size
            if "verb" == args.predict:
                loss = model.criteria_(pred["verb"], target[:, 0])
                acc = pred["verb"].argmax(1) == target[:, 0]
                print(pred['verb'].argmax(1)[:10])
                print(target[:, 0][:10])
                print(acc[:10])
                res['verb_loss'] += loss.detach().item()
                res['verb_correct'] += acc.sum()
            
            if "role" == args.predict:
                loss = model.criteria_(pred["role"].repeat(18, 1), target[:, 1::2].transpose(0, 1).flatten())
                res['role_loss'] += loss.detach().item()                
                for bidx in range(batch_size):
                    pred_argmax = pred["role"][bidx].argmax().cpu().item()
                    role_target = [n for n in target[bidx, 1::2].cpu().tolist() if n >= 0]
                    res["role_correct"] += pred_argmax in role_target

            if "noun" == args.predict:
                loss = model.criteria_(pred["noun"].repeat(18, 1), target[:, 2::2].transpose(0, 1).flatten())
                res['noun_loss'] += loss.detach().item()                
                for bidx in range(batch_size):
                    pred_argmax = pred["noun"][bidx].argmax().cpu().item()
                    noun_target = [n for n in target[bidx, 2::2].cpu().tolist() if n >= 0]
                    res["noun_correct"] += pred_argmax in noun_target
                #     noun_target = [n for n in target[bidx, 2::2] if n >= 0]
                #     sorted_pred = sorted(range(len(pred["noun"][bidx])), key=lambda k: pred["noun"][bidx][k], reverse=True)[:6]
                #     acc = np.array(sorted_pred) == np.array(noun_target)
                #     res['noun_correct'] += acc.sum()
                #     res['noun_all_correct'] += acc.all()
            
            if "frame" == args.predict:
                fr_target = torch.tensor([
                    model.encoding.fr_id[frozenset([fr for fr in trg if fr >= 0])]
                    for trg in target[:, 1::2].cpu().tolist()]).to(args.gpu)
                loss = model.criteria_(pred["frame"], fr_target)
                acc = pred["frame"].argmax(1) == fr_target
                res['frame_loss'] += loss.detach().item()
                res['frame_correct'] += acc.sum()

        epoch_res = {k: res[k]/res['total'] for k, v in res.items() if args.predict in k}
    return epoch_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', choices=["verb", "noun", "role", "frame"], default="role")
    parser.add_argument('--predict_top_k_noun', type=int, default=2000)
    parser.add_argument('--cnn-type', choices=['vgg'], default='vgg')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--encoding-file')
    parser.add_argument('--dataset-dir', default='./')
    parser.add_argument('--image-dir', default='./resized_256/')
    parser.add_argument('--output-dir', default='./test_backbone_result/')
    parser.add_argument('--num-epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    args = parser.parse_args()
    print(args)

    train_set = json.load(open(args.dataset_dir+"/train.json"))
    dev_set = json.load(open(args.dataset_dir+"/dev.json"))

    if args.encoding_file is None:
        encoder = imSituVerbRoleNounUNKEncoder(train_set, args.predict_top_k_noun)
        torch.save(encoder, args.output_dir + "/encoder")
    else:
        encoder = torch.load(args.encoding_file)

    if args.use_wandb:
        wandb.init(project='imSitu_YYS', name=args.predict, config=args)
    model = BackBone(encoder, args.cnn_type)

    dataset_train = imSituSituation(
        args.image_dir, train_set, encoder, model.train_preprocess())
    dataset_dev = imSituSituation(
        args.image_dir, dev_set, encoder, model.dev_preprocess())

    # dataset_train.ids = dataset_train.ids[:100]
    # dataset_dev.ids = dataset_dev.ids[:100]

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True)  # , num_workers = 3)
    dev_loader = torch.utils.data.DataLoader(
        dataset_dev, batch_size=args.batch_size, shuffle=False)  # , num_workers = 3)

    model.to(args.gpu)

    optimizer = optim.Adam(
        [{'params': model.cnn.parameters(), 'lr': 5e-5},
         {'params': model.linear_v.parameters(), 'lr': args.learning_rate},
         {'params': model.linear_n.parameters(), 'lr': args.learning_rate},
         {'params': model.linear_r.parameters(), 'lr': args.learning_rate},
         {'params': model.linear_f.parameters(), 'lr': args.learning_rate},
        ],
        lr = args.learning_rate , weight_decay = args.weight_decay)

    for epoch in range(args.num_epoch):
        eval_res = evaluation(model, dev_loader, args)
        print({k: '{:.4f}'.format(v) for k, v in eval_res.items() if k != 'total'})
        if args.use_wandb:
            wandb.log({"Eval {}".format(k): v for k, v in eval_res.items()})

        train_loop = tqdm(train_loader, total=len(train_loader))
        # train_loop = train_loader
        for idx, img, target in train_loop:
            train_res = train_batch(model, optimizer, img.to(args.gpu), target.to(args.gpu), args)
            if args.use_wandb:
                wandb.log({"Train {}".format(k): v for k, v in train_res.items()})

        torch.save(model.state_dict(), "{}/Epoch{:02d}".format(args.output_dir, epoch))
        if args.use_wandb:
            torch.save(model.state_dict(), "{}/Epoch{:02d}".format(wandb.run.dir, epoch))
        # break

    eval_res = evaluation(model, dev_loader, args)
    print({k: '{:.4f}'.format(v) for k, v in eval_res.items() if k != 'total'})
    if args.use_wandb:
        wandb.log({"Eval {}".format(k): v for k, v in eval_res.items()})
    
if __name__ == "__main__":
    main()