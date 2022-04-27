# Feel free to modifiy this file.
# It will only be used to verify the settings are correct
# modified from https://pytorch.org/docs/stable/data.html

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T
import utils
from engine import train_one_epoch, evaluate

from collections import OrderedDict
from dataset import UnlabeledDataset, LabeledDataset

# Rebuilt SwAV as the default model doesn't like to run correctly
class SwAV(nn.Module):
    def __init__(self):
        super(SwAV, self).__init__()
        self.model = torch.hub.load("facebookresearch/swav_ddp:main", "resnet50")
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        
    
    def forward(self, x):
        res = OrderedDict()
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        res["0"] = x
        x = self.model.layer2(x)
        res["1"] = x
        x = self.model.layer3(x)
        res["2"] = x
        x = self.model.layer4(x)
        res["3"] = x
        return res

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes, swav=False, eval=False):
    # model = torch.hub.load("facebookresearch/swav_ddp", "resnet50")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    if swav:
        model.backbone.body = SwAV()
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if eval:
        model.load_state_dict(torch.load("model-8.pth"))
        model.eval()

    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 100

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

    model = get_model(num_classes, swav=True, eval=True)
    model.to(device)

    evaluate(model, valid_loader, device=device)

if __name__ == "__main__":
    main()