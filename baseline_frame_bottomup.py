import os
import time
from torch import optim
import random as rand
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tvt
import math
from imsitu import imSituVerbFrameLocalRoleNounEncoder
from imsitu import imSituTensorEvaluation
from imsitu import imSituSituation
from imsitu import imSituSimpleImageFolder
from utils import initLinear
import json
import wandb


class VGGModified(nn.Module):
    def __init__(self):
        super(VGGModified, self).__init__()
        self.vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = self.vgg.features
        # self.classifier = nn.Sequential(
        # nn.Dropout(),
        self.lin1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout()
        self.lin2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout()

        initLinear(self.lin1)
        initLinear(self.lin2)

    def rep_size(self): return 1024

    def forward(self, x):
        return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))


class ResNetModifiedLarge(nn.Module):
    def __init__(self):
        super(ResNetModifiedLarge, self).__init__()
        self.resnet = tv.models.resnet101(pretrained=True)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*2048, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        # print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))


class ResNetModifiedMedium(nn.Module):
    def __init__(self):
        super(ResNetModifiedMedium, self).__init__()
        self.resnet = tv.models.resnet50(pretrained=True)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*2048, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        # print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))


class ResNetModifiedSmall(nn.Module):
    def __init__(self):
        super(ResNetModifiedSmall, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=True)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*512, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def base_size(self): return 512
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))


class BaselineFrameBottomUp(nn.Module):
    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    # prediction type can be "max_max" or "max_marginal"
    def __init__(self, encoding, node_hidden_layer,
                 prediction_type="max_max", device_array=[0], cnn_type="resnet_101"):
        super(BaselineFrameBottomUp, self).__init__()

        self.normalize = tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.RandomCrop(224),
            # tv.transforms.RandomRotation(10),
            # tv.transforms.RandomResizedCrop(224),
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

        self.broadcast = []
        self.encoding = encoding
        self.node_hidden_layer = node_hidden_layer
        self.prediction_type = prediction_type
        self.n_verbs = encoding.n_verbs()
        # cnn
        print(cnn_type)
        if cnn_type == "resnet_101":
            self.cnn = ResNetModifiedLarge()
        elif cnn_type == "resnet_50":
            self.cnn = ResNetModifiedMedium()
        elif cnn_type == "resnet_34":
            self.cnn = ResNetModifiedSmall()
        else:
            print("unknown base network")
            exit()
        self.rep_size = self.cnn.rep_size()

        # create the mapping for grouping the roles back to the verbs later
        max_roles = encoding.max_roles()

        # need a list that is nverbs by 6
        self.v_r = [0 for i in range(0, self.encoding.n_verbs()*max_roles)]

        # and we need to compute the position of the corresponding roles, and pad with the 0 symbol
        for v in self.encoding.id_v:
            offset = max_roles*v
            # stored in role order
            vf = self.encoding.v_f[v]
            for f in vf:
                roles = self.encoding.f_r[f]
                k = 0
                for r in roles:
                    # add one to account of the 0th element being the padding
                    self.v_r[offset + k] = r + 1
                    k += 1
                # pad
                while k < max_roles:
                    self.v_r[offset + k] = 0
                    k += 1

        for g in device_array:
            self.broadcast.append(
                Variable(torch.LongTensor(self.v_r).cuda(g)))

        hidden_layer = [nn.Identity()] if len(
            self.node_hidden_layer) == 0 else []
        node_size = self.rep_size
        for node_size_ in self.node_hidden_layer:
            hidden_layer += [nn.Linear(node_size, node_size_),
                             nn.ReLU(), nn.Dropout(.5)]
            node_size = node_size_

        self.frame_node = nn.ModuleList([
            nn.Sequential(*hidden_layer)
            for f in range(self.encoding.n_frames())])

        # verb potential
        self.linear_fv = nn.ModuleList([
            nn.Linear(node_size, len(fv))
            for f, fv in self.encoding.f_v.items()])
        self.total_fv = 0
        for _, fv in self.encoding.f_v.items():
            self.total_fv += len(fv)

        # role-noun potentials
        self.linear_rn = nn.ModuleList([
            nn.Linear(node_size, len(rn))
            for r, rn in self.encoding.r_id_n.items()])
        self.total_rn = 0
        for r, rn in self.encoding.r_id_n.items():
            self.total_rn += len(rn)

        print("total fv: {0}, total rn : {1}, encoding rn : {2}".format(
            self.total_fv, self.total_rn, encoding.n_rolenoun()))

        # initilize everything
        for _l in self.linear_fv:
            initLinear(_l)
        for _l in self.linear_rn:
            initLinear(_l)

    def forward_features(self, images):
        return self.cnn(images)

    def forward(self, image):
        batch_size = image.size()[0]

        rep = self.cnn(image)
        frame_rep = [frame_node(rep)
                     for frame_node in self.frame_node]

        frn_potential = []
        frn_marginal = []
        fr_max = []
        fr_maxi = []
        for f, f_rep in enumerate(frame_rep):
            for r in self.encoding.f_r[f]:
                frn_group = self.linear_rn[r]
                _frn_potential = frn_group(f_rep)
                frn_potential.append(_frn_potential)

                _frn_marginal = _frn_potential.logsumexp(1, keepdim=True)
                frn_marginal.append(_frn_marginal)

                _fr_max, _fr_maxi = _frn_potential.max(1, keepdim=True)
                fr_maxi.append(_fr_maxi)
                fr_max.append(_fr_max)

        # concat role groups with the padding symbol
        zeros = Variable(torch.zeros(batch_size, 1))  # this is the padding
        zerosi = Variable(torch.LongTensor(batch_size, 1).zero_())
        frn_marginal.insert(0, zeros.to(rep.device))
        fr_max.insert(0, zeros.to(rep.device))
        fr_maxi.insert(0, zerosi.to(rep.device))

        frn_marginal = torch.cat(frn_marginal, 1)
        fr_max = torch.cat(fr_max, 1)
        fr_maxi = torch.cat(fr_maxi, 1)

        v_r = self.broadcast[torch.cuda.current_device()]
        frn_marginal_grouped = frn_marginal.index_select(1, v_r).view(
            batch_size, self.n_verbs, self.encoding.max_roles())
        fr_max_grouped = fr_max.index_select(1, v_r).view(
            batch_size, self.n_verbs, self.encoding.max_roles())
        fr_maxi_grouped = fr_maxi.index_select(1, v_r).view(
            batch_size, self.n_verbs, self.encoding.max_roles())

        fv_potential = torch.full((batch_size, self.encoding.n_frames(), self.n_verbs), float('-inf')
                                  ).to(frn_marginal.device)
        for f, (fv_group, f_rep) in enumerate(zip(self.linear_fv, frame_rep)):
            _fv_potential = fv_group(f_rep)
            v_idx = torch.LongTensor(
                self.encoding.f_v[f]).to(_fv_potential.device)
            fv_potential[:, f, v_idx] = _fv_potential
        v_potential = fv_potential.logsumexp(dim=1)

        marginal = frn_marginal_grouped.sum(2).view(
            batch_size, self.n_verbs) + v_potential

        norm = marginal.logsumexp(1)

        _max = fr_max_grouped.sum(2).view(
            batch_size, self.n_verbs) + v_potential  # these are the scores

        if self.prediction_type == "max_max":
            rv = (rep, v_potential, frn_potential,
                  norm, _max, fr_maxi_grouped)
        elif self.prediction_type == "max_marginal":
            rv = (rep, frn_marginal, frn_potential,
                  norm, v_potential, fr_maxi_grouped)
        else:
            print("unkown inference type")
            rv = ()
        return rv

    def logsumexp_nx_ny_xy(self, x, y):
        if x > y:
            return torch.log(torch.exp(y-x) + 1 - torch.exp(y) + 1e-8) + x
        else:
            return torch.log(torch.exp(x-y) + 1 - torch.exp(x) + 1e-8) + y

    def sum_loss(self, v_potential, rn_potential, norm, situations, n_refs):
        batch_size = v_potential.size()[0]
        mr = self.encoding.max_roles()
        for i in range(0, batch_size):
            _norm = norm[i]
            _v = v_potential[i]
            _rn = []
            _ref = situations[i]
            for pot in rn_potential:
                _rn.append(pot[i])
            for ref in range(0, n_refs):
                v = _ref[0]
                pots = _v[v]
                for (pos, r) in enumerate(self.encoding.v_r[v.item()]):
                    pots = pots + _rn[r][_ref[1 + 2*mr*ref + 2*pos + 1]]
                if pots.data[0] > _norm.data[0]:
                    print("inference error")
                    print(pots)
                    print(_norm)
                if i == 0 and ref == 0:
                    loss = pots-_norm
                else:
                    loss = loss + pots - _norm
        return -loss/(batch_size*n_refs)

    def mil_loss(self, v_potential, frn_potential, norm,  situations, n_refs):
        batch_size = v_potential.size()[0]
        mr = self.encoding.max_roles()
        for i in range(0, batch_size):
            _norm = norm[i]
            _v = v_potential[i]
            _frn = []
            _ref = situations[i]
            for pot in frn_potential:
                _frn.append(pot[i])
            for ref in range(0, n_refs):
                v = _ref[0]
                pots = _v[v]
                vf = self.encoding.v_f[v.item()]
                for f in vf:
                    fr = sum([len(self.encoding.f_r[_f]) for _f in range(f)])
                    for (pos, r) in enumerate(self.encoding.f_r[f]):
                        pots = pots + \
                            _frn[fr+pos][_ref[1 + 2*mr*ref + 2*pos + 1]]
                if pots.item() > _norm.item():
                    print("inference error")
                    print(pots)
                    print(_norm)
                if ref == 0:
                    _tot = pots-_norm
                else:
                    _tot = self.logsumexp_nx_ny_xy(_tot, pots-_norm)
            if i == 0:
                loss = _tot
            else:
                loss = loss + _tot
        return -loss/batch_size


def format_dict(d, s, p):
    rv = ""
    for (k, v) in d.items():
        if len(rv) > 0:
            rv += " , "
        rv += p+str(k) + ":" + s.format(v*100)
    return rv


def predict_human_readable(dataset_loader, simple_dataset, encoder, model, outdir, top_k):
    model.eval()
    print("predicting...")
    mx = len(dataset_loader)
    with torch.no_grad():
        for i, (input, index) in enumerate(dataset_loader):
            print("{}/{} batches".format(i+1, mx))
            input_var = input.cuda()
            _, _, _, _, scores, predictions = model.forward(input_var)
            # (s_sorted, idx) = torch.sort(scores, 1, True)
            human = encoder.to_situation(predictions)
            (b, p, d) = predictions.size()
            for _b in range(0, b):
                items = []
                offset = _b * p
                for _p in range(0, p):
                    items.append(human[offset + _p])
                    items[-1]["score"] = scores.data[_b][_p].item()
                items = sorted(items, key=lambda x: -x["score"])[:top_k]
                name = simple_dataset.images[index[_b][0]].split(".")[:-1]
                name.append("predictions")
                outfile = outdir + ".".join(name)
                json.dump(items, open(outfile, "w"))


def compute_features(dataset_loader, simple_dataset,  model, outdir):
    model.eval()
    print("computing features...")
    mx = len(dataset_loader)
    with torch.no_grad():
        for i, (input, index) in enumerate(dataset_loader):
            print("{}/{} batches\r".format(i+1, mx)),
            input_var = input.cuda()
            features = model.forward_features(input_var).cpu().data
            b = index.size()[0]
            for _b in range(0, b):
                name = simple_dataset.images[index[_b][0]].split(".")[:-1]
                name.append("features")
                outfile = outdir + ".".join(name)
                torch.save(features[_b], outfile)
    print("\ndone.")


def eval_model(dataset_loader, encoding, model, device):
    model.eval()
    print("evaluating model...")
    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)

    mx = len(dataset_loader)
    with torch.no_grad():
        for i, (index, input, target) in enumerate(dataset_loader):
            print("{}/{} batches\r".format(i+1, mx)),
            input_var = input.to(device)
            target_var = target.to(device)
            _, _, _, _, scores, predictions = model.forward(input_var)
            (s_sorted, idx) = torch.sort(scores, 1, True)
            top1.add_point(target, predictions.data, idx.data)
            top5.add_point(target, predictions.data, idx.data)

    print("\ndone.")
    return (top1, top5)


def train_model(max_epoch, eval_frequency, train_loader, dev_loader, model, encoding, optimizer, save_dir, device_array, args, timing=False):
    if args.use_wandb:
        wandb.init(project='imSitu_YYS3', name='Frame_BottomUp', config=args)
    model.train()

    time_all = time.time()

    pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)
    loss_total = 0
    print_freq = 10
    total_steps = 0
    avg_scores = []

    for k in range(0, max_epoch):
        for i, (index, input, target) in enumerate(train_loader):
            total_steps += 1

            t0 = time.time()
            t1 = time.time()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            (_, v, rn, norm, scores, predictions) = pmodel(input_var)
            (s_sorted, idx) = torch.sort(scores, 1, True)
            # print norm
            if timing:
                print("forward time = {}".format(time.time() - t1))
            optimizer.zero_grad()
            t1 = time.time()
            loss = model.mil_loss(v, rn, norm, target, 3)
            if timing:
                print("loss time = {}".format(time.time() - t1))
            t1 = time.time()
            loss.backward()
            # print loss
            if timing:
                print("backward time = {}".format(time.time() - t1))
            optimizer.step()
            loss_total += loss.item()
            # score situation
            t2 = time.time()
            top1.add_point(target, predictions.data, idx.data)
            top5.add_point(target, predictions.data, idx.data)

            if timing:
                print("eval time = {}".format(time.time() - t2))
            if timing:
                print("batch time = {}".format(time.time() - t0))
            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results()
                top5_a = top5.get_average_results()
                print("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}, batch time = {:.2f}".format(total_steps-1, k, i, format_dict(top1_a, "{:.2f}", "1-"), format_dict(
                    top5_a, "{:.2f}", "5-"), loss.item(), loss_total / ((total_steps-1) % eval_frequency), (time.time() - time_all) / ((total_steps-1) % eval_frequency)))
                if args.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/avg_loss': loss_total / ((total_steps-1) % eval_frequency),
                    }, step=total_steps)
                    wandb.log({'train/top1-{}'.format(k): v for k, v in top1_a.items()},
                              step=total_steps)
                    wandb.log({'train/top5-{}'.format(k): v for k, v in top5_a.items()},
                              step=total_steps)
            if total_steps % eval_frequency == 0:
                print("eval...")
                etime = time.time()
                (top1, top5) = eval_model(dev_loader, encoding,
                                          pmodel, torch.device(device_array[0]))
                model.train()
                print("... done after {:.2f} s".format(time.time() - etime))
                top1_a = top1.get_average_results()
                top5_a = top5.get_average_results()

                avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + \
                    top5_a["value"] + top5_a["value-all"] + \
                    top5_a["value*"] + top5_a["value-all*"]
                avg_score /= 8

                print("Dev {} average :{:.2f} {} {}".format(total_steps-1, avg_score*100,
                                                            format_dict(top1_a, "{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-")))
                if args.use_wandb:
                    wandb.log({
                        'eval/avg_score': avg_score,
                    }, step=total_steps)
                    wandb.log({'eval/top1-{}'.format(k): v for k, v in top1_a.items()},
                              step=total_steps)
                    wandb.log({'eval/top5-{}'.format(k): v for k, v in top5_a.items()},
                              step=total_steps)

                avg_scores.append(avg_score)
                maxv = max(avg_scores)

                if maxv == avg_scores[-1]:
                    torch.save(model.state_dict(), save_dir +
                               "/{0}.model".format(maxv))
                    print("new best model saved! {0}".format(maxv))

                top1 = imSituTensorEvaluation(1, 3, encoding)
                top5 = imSituTensorEvaluation(5, 3, encoding)
                loss_total = 0
                time_all = time.time()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--command",
                        choices=["train", "eval", "predict", "features"], required=True)
    parser.add_argument("--output_dir",
                        help="location to put output, such as models, features, predictions")
    parser.add_argument("--image_dir", default="./resized_256",
                        help="location of images to process")
    parser.add_argument("--dataset_dir", default="./",
                        help="location of train.json, dev.json, ect.")
    parser.add_argument("--weights_file", help="the model to start from")
    parser.add_argument("--encoding_file",
                        help="a file corresponding to the encoder")
    parser.add_argument("--cnn_type", choices=["resnet_34", "resnet_50", "resnet_101"],
                        default="resnet_101", help="the cnn to initilize")
    parser.add_argument("--node_hidden_layer", type=int, nargs='*',
                        default=[32], help="the role node hidden layer sizes")
    parser.add_argument("--batch_size", default=64,
                        help="batch size for training", type=int)
    parser.add_argument("--learning_rate", default=1e-5,
                        help="learning rate for ADAM", type=float)
    parser.add_argument("--weight_decay", default=5e-4,
                        help="learning rate decay for ADAM", type=float)
    parser.add_argument("--eval_frequency", default=500,
                        help="evaluate on dev set every N training steps", type=int)
    parser.add_argument("--training_epochs", default=20,
                        help="total number of training epochs", type=int)
    parser.add_argument("--eval_file", default="dev.json",
                        help="the dataset file to evaluate on, ex. dev.json test.json")
    parser.add_argument("--top_k", default="10", type=int,
                        help="topk to use for writing predictions to file")
    parser.add_argument("--device_array", nargs='+', type=int, default=[0])
    parser.add_argument("--use_wandb", action='store_true')

    args = parser.parse_args()
    device = torch.device(args.device_array[0])
    if args.command == "train":
        print("command = training")
        train_set = json.load(open(args.dataset_dir+"/train.json"))
        dev_set = json.load(open(args.dataset_dir+"/dev.json"))

        if args.encoding_file is None:
            encoder = imSituVerbFrameLocalRoleNounEncoder(train_set)
            torch.save(encoder, args.output_dir + "/encoder")
        else:
            encoder = torch.load(args.encoding_file)

        model = BaselineFrameBottomUp(encoder, cnn_type=args.cnn_type, node_hidden_layer=args.node_hidden_layer,
                                      device_array=args.device_array)

        if args.weights_file is not None:
            model.load_state_dict(torch.load(args.weights_file))

        dataset_train = imSituSituation(
            args.image_dir, train_set, encoder, model.train_preprocess())
        dataset_dev = imSituSituation(
            args.image_dir, dev_set, encoder, model.dev_preprocess())

        batch_size = args.batch_size*len(args.device_array)

        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True)  # , num_workers = 3)
        dev_loader = torch.utils.data.DataLoader(
            dataset_dev, batch_size=batch_size, shuffle=True)  # , num_workers = 3)

        # model.cuda(args.device_array[0])
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        train_model(args.training_epochs, args.eval_frequency, train_loader,
                    dev_loader, model, encoder, optimizer, args.output_dir, args.device_array, args)

    elif args.command == "eval":
        print("command = evaluating")
        eval_file = json.load(open(args.dataset_dir + "/" + args.eval_file))

        if args.encoding_file is None:
            print("expecting encoder file to run evaluation")
            exit()
        else:
            encoder = torch.load(args.encoding_file)
        print("creating model...")
        model = BaselineFrameBottomUp(encoder, cnn_type=args.cnn_type)

        if args.weights_file is None:
            print("expecting weight file to run features")
            exit()

        print("loading model weights...")
        model.load_state_dict(torch.load(args.weights_file))
        # model.cuda()

        dataset = imSituSituation(
            args.image_dir, eval_file, encoder, model.dev_preprocess())
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

        (top1, top5) = eval_model(loader, encoder, model)
        top1_a = top1.get_average_results()
        top5_a = top5.get_average_results()

        avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + \
            top5_a["value"] + top5_a["value-all"] + \
            top5_a["value*"] + top5_a["value-all*"]
        avg_score /= 8

        print("Average :{:.2f} {} {}".format(
            avg_score*100, format_dict(top1_a, "{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-")))

    elif args.command == "features":
        print("command = features")
        if args.encoding_file is None:
            print("expecting encoder file to run features")
            exit()
        else:
            encoder = torch.load(args.encoding_file)

        print("creating model...")
        model = BaselineFrameBottomUp(encoder, cnn_type=args.cnn_type)

        if args.weights_file is None:
            print("expecting weight file to run features")
            exit()

        print("loading model weights...")
        model.load_state_dict(torch.load(args.weights_file))
        model.cuda()

        folder_dataset = imSituSimpleImageFolder(
            args.image_dir, model.dev_preprocess())
        image_loader = torch.utils.data.DataLoader(
            folder_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)

        compute_features(image_loader, folder_dataset, model, args.output_dir)

    elif args.command == "predict":
        print("command = predict")
        if args.encoding_file is None:
            print("expecting encoder file to run features")
            exit()
        else:
            encoder = torch.load(args.encoding_file)

        print("creating model...")
        model = BaselineFrameBottomUp(encoder, cnn_type=args.cnn_type)

        if args.weights_file is None:
            print("expecting weight file to run features")
            exit()

        print("loading model weights...")
        model.load_state_dict(torch.load(args.weights_file))
        model.cuda()

        folder_dataset = imSituSimpleImageFolder(
            args.image_dir, model.dev_preprocess())
        image_loader = torch.utils.data.DataLoader(
            folder_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)

        predict_human_readable(image_loader, folder_dataset, encoder,
                               model, args.output_dir, args.top_k)


if __name__ == "__main__":
    main()
