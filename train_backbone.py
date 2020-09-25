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
from utils import vgg_bn_modified as VGGBNModified
import wandb


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

        sorted_n_count = sorted(self.n_count.items(),
                                key=lambda x: x[1], reverse=True)
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


class Backbone(nn.Module):
    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    def __init__(self, encoding, cnn_type, num_class=10):
        super(Backbone, self).__init__()
        self.encoding = encoding

        if cnn_type == "vgg":
            self.cnn = VGGModified()
        elif cnn_type == "vgg_bn":
            self.cnn = VGGBNModified()
        self.rep_size = self.cnn.rep_size()

        self.classifier = nn.Sequential(
            nn.Linear(self.rep_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_class),
        )

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

    def forward(self, image):
        rep = self.cnn(image)
        return self.classifier(rep)


def situ2target(situ, predict, encoder):
    if predict == 'verb':
        target = situ[:, 0]
    elif predict == 'noun':
        target = torch.zeros(
            (situ.shape[0], encoder.n_nouns()+1), device=situ.device)
        target.scatter_(1, situ[:, 2::2]+1,
                        torch.ones_like(situ[:, 2::2]).float())
        target = target[:, 1:]
    elif predict == 'role':
        target = torch.zeros(
            (situ.shape[0], encoder.n_roles()+1), device=situ.device)
        target.scatter_(1, situ[:, 1::2]+1,
                        torch.ones_like(situ[:, 1::2]).float())
        target = target[:, 1:]
    elif predict == 'frame':
        target = torch.tensor([
            encoder.fr_id[frozenset([fr for fr in trg if fr >= 0])]
            for trg in situ[:, 1::2].cpu().tolist()]).to(situ.device)
    return target


def train_batch(model, train_criteria, optimizer, input_, situ, args):
    model.train()

    optimizer.zero_grad()
    pred = model(input_)
    target = situ2target(situ.to(pred.device), args.predict, model.encoding)
    loss = train_criteria(pred, target)
    loss.backward()
    optimizer.step()

    return {'{}_loss'.format(args.predict): loss.detach().item()}


def eval_batch(model, eval_criteria, eval_metric, input_, situ, args):
    res = {}
    model.eval()
    with torch.no_grad():
        res['batch_size'] = int(input_.shape[0])
        pred = model(input_)

        target = situ2target(situ.to(pred.device),
                             args.predict, model.encoding)
        loss = eval_criteria(pred, target)
        res['{}_loss'.format(args.predict)] = loss.detach().item()

        metric = eval_metric(pred, target)
        for k in metric:
            res['{}_{}'.format(args.predict, k)] = \
                metric[k].sum().item()

    return res


def evaluation(model, eval_criteria, eval_metric, eval_loader, args):
    model.eval()
    with torch.no_grad():
        eval_loop = tqdm(eval_loader, total=len(eval_loader))
        # eval_loop = eval_loader
        columns = ["Target vid", "Target verb",
                   "Top 10 Pred vid", "Top 10 Pred verb"]
        pd_res = pd.DataFrame(columns=columns)
        res = {"total": 0,
               "verb_loss": 0., "verb_correct": 0,
               "noun_loss": 0., "noun_IoU": 0,  # "noun_all_correct": 0,
               "role_loss": 0., "role_IoU": 0,  # "role_all_correct": 0,
               "frame_loss": 0., "frame_correct": 0}
        for idx, img, situ in eval_loop:
            batch_size = img.shape[0]
            input_ = img.to(args.gpu)
            pred = model(input_)

            target = situ2target(situ.to(pred.device),
                                 args.predict, model.encoding)
            loss = eval_criteria(pred, target)
            res['{}_loss'.format(args.predict)] += loss.detach().item()

            metric = eval_metric(pred, target)
            for k in metric:
                res['{}_{}'.format(args.predict, k)] += metric[k].sum().item()

            res['total'] += batch_size

        epoch_res = {k: res[k]/res['total']
                     for k, v in res.items() if args.predict in k}
    return epoch_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--predict', choices=["verb", "noun", "role", "frame"], default="role")
    parser.add_argument('--predict_top_k_noun', type=int, default=2000)
    parser.add_argument('--cnn-type', choices=['vgg', 'vgg_bn'], default='vgg')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--encoding-file')
    parser.add_argument('--dataset-dir', default='./')
    parser.add_argument('--image-dir', default='./resized_256/')
    parser.add_argument('--output-dir', default='./train_backbone_result/')
    parser.add_argument('--num-epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    args = parser.parse_args()
    print(args)

    if args.use_wandb:
        wandb.init(project='imSitu_YYS2', name=args.predict, config=args)

    train_set = json.load(open(args.dataset_dir+"/train.json"))
    dev_set = json.load(open(args.dataset_dir+"/dev.json"))

    if args.encoding_file is None:
        encoder = imSituVerbRoleNounUNKEncoder(
            train_set, args.predict_top_k_noun)
        if args.use_wandb:
            torch.save(encoder, wandb.run.dir + "/encoder")
        else:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            torch.save(encoder, args.output_dir + "/encoder")
    else:
        encoder = torch.load(args.encoding_file)

    if args.predict == 'verb':
        num_class = encoder.n_verbs()
    elif args.predict == 'noun':
        num_class = encoder.n_nouns()
    elif args.predict == 'role':
        num_class = encoder.n_roles()
    elif args.predict == 'frame':
        num_class = encoder.n_frames()
    model = Backbone(encoder, args.cnn_type, num_class)

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
         {'params': model.classifier.parameters(), 'lr': args.learning_rate},
         ],
        lr=args.learning_rate, weight_decay=args.weight_decay)

    def correct_metric(pred, target):
        correct = pred.argmax(1) == target
        return {'correct': correct}

    def IoU_metric(pred, target):
        IoU = torch.sum((pred >= 0.5) * target.bool()) / \
            torch.sum((pred >= 0.5) + target.bool())
        IoU /= float(pred.shape[1])
        return {'IoU': IoU}

    if args.predict in ["verb", "frame"]:
        train_criteria = nn.CrossEntropyLoss()
        eval_criteria = nn.CrossEntropyLoss(reduction='sum')
        eval_metric = correct_metric
    elif args.predict in ["noun", "role"]:
        train_criteria = nn.BCEWithLogitsLoss()
        eval_criteria = nn.BCEWithLogitsLoss(reduction='sum')
        eval_metric = IoU_metric
    else:
        assert False

    for epoch in range(args.num_epoch):
        eval_res = evaluation(model, eval_criteria,
                              eval_metric, dev_loader, args)
        print({k: '{:.4f}'.format(v)
               for k, v in eval_res.items() if k != 'total'})
        if args.use_wandb:
            wandb.log({"Eval {}".format(k): v for k, v in eval_res.items()},
                      step=epoch*len(train_loader))

        train_loop = tqdm(enumerate(train_loader), total=len(train_loader))
        # train_loop = train_loader
        running_eval_res = {'total': 0}
        for batch_idx, (idx, img, situ) in train_loop:
            train_res = train_batch(
                model, train_criteria, optimizer,
                img.to(args.gpu), situ.to(args.gpu), args)
            eval_res = eval_batch(
                model, eval_criteria, eval_metric,
                img.to(args.gpu), situ.to(args.gpu), args)

            running_eval_res['total'] += eval_res['batch_size']
            for k, v in eval_res.items():
                if 'batch_size' != k:
                    if k not in running_eval_res:
                        running_eval_res[k] = [0.]*len(train_loader)
                    running_eval_res[k][batch_idx] = v
            if args.use_wandb:
                wandb.log({"Train {}".format(k): v
                           for k, v in train_res.items()},
                          step=epoch*len(train_loader)+batch_idx)
                wandb.log({"Train {}".format(k): sum(v) / min(running_eval_res['total'], len(train_loader.dataset))
                           for k, v in running_eval_res.items() if 'total' != k},
                          step=epoch*len(train_loader)+batch_idx)

        if args.use_wandb:
            torch.save(model.state_dict(),
                       "{}/Epoch{:02d}".format(wandb.run.dir, epoch))
        else:
            torch.save(model.state_dict(),
                       "{}/Epoch{:02d}".format(args.output_dir, epoch))

        # break

    eval_res = evaluation(model, eval_criteria, eval_metric, dev_loader, args)
    print({k: '{:.4f}'.format(v) for k, v in eval_res.items() if k != 'total'})
    if args.use_wandb:
        wandb.log({"Eval {}".format(k): v for k, v in eval_res.items()},
                  step=args.num_epoch*len(train_loader))


if __name__ == "__main__":
    main()
