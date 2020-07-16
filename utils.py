import json

import torch
import torchvision as tv
from torch import nn

from imsitu import (imSituVerbRoleLocalNounEncoder,
                    imSituTensorEvaluation,
                    imSituSituation,
                    imSituSimpleImageFolder,
                    )


class vgg_modified(nn.Module):
    def __init__(self):
        super(vgg_modified,self).__init__()
        self.vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = self.vgg.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            )

    def rep_size(self):
        return 1024

    def forward(self,x):
        x = self.vgg_features(x)
        x = x.view(-1, 512*7*7)
        x = self.classifier(x)
        return x

def make_resnet_feature_extractor(resnet):
    return nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        nn.Dropout2d(.5),
    )

class resnet_modified_large(nn.Module):
    def __init__(self):
        super(resnet_modified_large, self).__init__()
        self.resnet = tv.models.resnet101(pretrained=True)
        self.feature_extractor = make_resnet_feature_extractor(self.resnet)
        self.classifier = nn.Sequential(
            nn.Linear(7*7*2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(.5),
        )


    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 7*7*self.base_size())
        x = self.classifier(x)
        return x

class resnet_modified_medium(nn.Module):
    def __init__(self):
        super(resnet_modified_medium, self).__init__()
        self.resnet = tv.models.resnet50(pretrained=True)
        self.feature_extractor = make_resnet_feature_extractor(self.resnet)
        self.classifier = nn.Sequential(
            nn.Linear(7*7*2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(.5),
        )

    def _make_feature_extractor(self, resnet):
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.Dropout2d(.5),
        )

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 7*7*self.base_size())
        x = self.classifier(x)
        return x 
 
class resnet_modified_small(nn.Module):
    def __init__(self):
        super(resnet_modified_small, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=True)
        self.feature_extractor = make_resnet_feature_extractor(self.resnet)
        self.classifier = nn.Sequential(
            nn.Linear(7*7*2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(.5),
        )

    def _make_feature_extractor(self, resnet):
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.Dropout2d(.5),
        )

    def base_size(self): return 512
    def rep_size(self): return 1024
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 7*7*self.base_size())
        x = self.classifier(x)
        return x 

def format_dict(d, s, p):
    rv = ""
    for (k,v) in d.items():
        if len(rv) > 0: rv += " , "
        rv+=p+str(k) + ":" + s.format(v*100)
    return rv

def predict_human_readable (dataset_loader, simple_dataset,  model, outdir, top_k):
    model.eval()  
    print("predicting...")
    mx = len(dataset_loader) 
    with torch.no_grad():
        for i, (input, index) in enumerate(dataset_loader):
            print ("{}/{} batches".format(i+1,mx))
            input_var = torch.autograd.Variable(input.cuda(), volatile = True)
            (scores,predictions)  = model.forward_max(input_var)
            #(s_sorted, idx) = torch.sort(scores, 1, True)
            human = model.encoding.to_situation(predictions)
            (b,p,d) = predictions.size()
            for _b in range(0,b):
                items = []
                offset = _b *p
                for _p in range(0, p):
                    items.append(human[offset + _p])
                    items[-1]["score"] = scores.data[_b][_p].item()
                items = sorted(items, key = lambda x: -x["score"])[:top_k]
                name = simple_dataset.images[index[_b][0]].split(".")[:-1]
                name.append("predictions")
                outfile = outdir + ".".join(name)
                json.dump(items,open(outfile,"w"))
