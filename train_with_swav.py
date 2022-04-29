# -----------------------------------------------------------
# Load swav pretrained backbone and train FasterRCNN using supervised dataset
#
#   note: only work for resnet50
# -----------------------------------------------------------
import argparse
from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
import utils
from engine import train_one_epoch, evaluate
from dataset import UnlabeledDataset, LabeledDataset
from typing import Dict
from torchvision.models.detection.transform import GeneralizedRCNNTransform
# TODO: Might need to be modified depends on the structure of our project files
import swav_simplified.src.resnet50 as resnet_models
import sys
import time

parser = argparse.ArgumentParser(description="Evaluate models: Fine-tuning with labeled dataset")
#########################
#### swav parameters ####
#########################
parser.add_argument("--pretrained_hub", type=bool, default=True,
                    help="if backbone downloaded from Facebook hub")
parser.add_argument("--ckp_path", type=str, default="./swav_res/ckp-30.pth",
                    help="path to imagenet")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")

#############################
#### training parameters ####
#############################
parser.add_argument("--mode", choices=['train', 'eval', 'resume'], 
                    default='train', type=str, help="Choose action.") #TODO: add resume impl
parser.add_argument("--epochs", default=30, type=int,
                    help="number of total epochs to run")
parser.add_argument("--eval_freq", default=5, type=int,
                    help="Eval the model periodically")
parser.add_argument("--checkpoint_freq", type=int, default=3,
                    help="Save the model periodically")
parser.add_argument("--arch", choices=['resnet50', 'resnet18', 'resnet34'],
                    default='resnet50', type=str, help="Architecture")
##########################
#### other parameters ####
##########################
parser.add_argument("--workers", default=2, type=int,
                    help="number of data loading workers")
parser.add_argument("--data_path", type=str, default="../../data/labeled",
                    help="path to imagenet")
parser.add_argument("--checkpoint_path", type=str, default="./back_up",
                    help="path to save back up model's weights")
parser.add_argument("--checkpoint_file", type=str, default="",
                    help="name of saved model")
parser.add_argument("--swav_path", type=str, default="./swav_results",
                    help="path to swav checkpoints")

parser.add_argument("--debug", type=bool, default=False,
                    help="DEBUG architecture of model")

class CustomizedBoxHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.ln1 = nn.LayerNorm(representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.ln2 = nn.LayerNorm(representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = self.fc6(x)
        x = F.relu(self.ln1(x))
        x = self.fc7(x)
        x = F.relu(self.ln2(x))

        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    https://github.com/pytorch/vision/blob/de31e4b8bf9b4a7e0668d19059a5ac4760dceee1/torchvision/models/_utils.py#L13
    
    Examples::
        >>> m = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
        

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def load_pretrained_swav(ckp_path):
    model = resnet_models.__dict__[args.arch](
        hidden_mlp=args.hidden_mlp, 
        output_dim=args.output_dim, 
        nmb_prototypes=args.nmb_prototypes
    )
    if os.path.isfile(ckp_path):
        state_dict = torch.load(ckp_path, map_location="cuda")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            
            if k not in list(state_dict):
                print('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                print('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print("Load pretrained model with msg: {}".format(msg))
    else:
        print("No pretrained weights found => training from random weights")
    return model


def get_model(num_classes, pretrained_hub, returned_layers=None):
    if args.mode == 'eval' or args.mode == 'resume':
        model = torch.load(os.path.join(args.checkpoint_path, args.checkpoint_file))
        model.eval()
        return model

    if pretrained_hub:
        backbone = torch.hub.load("facebookresearch/swav", "resnet50")
        # backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        print("Load pretrained swav backbone from Facebook hub")
    else:
        if not os.path.isdir(args.swav_path):
            raise Exception('Cannot find checkpoint path')
        else:
            backbone = load_pretrained_swav(args.swav_path, hidden_mlp=args.hidden_mlp, output_dim=args.output_dim, num_prototype=args.nmb_prototypes)
            backbone.padding = nn.Identity() #TODO: need to double check this
            backbone.projection_head = nn.Identity()
            backbone.prototypes = nn.Identity()
            print("Load pretrained swav backbone from OUR checkpoint")
    backbone.avgpool = nn.Identity()
    backbone.fc = nn.Identity()

    # --------------------Map resnet backbone---------------------
    # TODO: only for resnet50
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    new_back_bone = IntermediateLayerGetter(backbone, return_layers=return_layers)


    # --------------------standard demo thing---------------------

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    # Add normalize layer based on labeled dataset
    min_size, max_size = 800, 1333
    image_mean, image_std = [0.4697, 0.4517, 0.3954], [0.2305, 0.2258, 0.2261]
    norm_layer = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    model.transform = norm_layer

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=in_features, out_features=num_classes)
    # model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=in_features, out_features=num_classes * 4)

    # ---------------------------------------------------------
    # Change backbone
    # ---------------------------------------------------------
    # model.backbone.body.load_state_dict(new_back_bone.state_dict())
    model.backbone.body = new_back_bone
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.momentum = 0.01
            mod.running_var.fill_(1)
            mod.running_mean.zero_()

    for m in model.parameters():
        m.requires_grad = True


    # ---------------------------------------------------------
    # Change box_head
    # ---------------------------------------------------------
    out_channels = model.backbone.out_channels
    resolution = model.roi_heads.box_roi_pool.output_size[0]
    representation_size = 1024
    new_box_head = CustomizedBoxHead(out_channels * resolution ** 2, representation_size)
    model.roi_heads.box_head = new_box_head


    if args.debug:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'TRAIN: {name}')
            else:
                print(f'FROZEN: {name}')

        for name, child in model.named_children():
            print(f'name is: {name}')
            print(f'module is: {child}')
        # print(model.backbone.body.layer4[0].bn1.running_var)
        # print(model.backbone.body.layer4[0].bn1.running_mean)
        print("DONE")
        sys.stdout.flush()
        time.sleep(3)
    # raise NotImplementedError

    return model

def main():
    global args
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not os.path.isdir(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    pd_train_header = [
        "Epoch", "loss", "loss_classifier", 
        "loss_box_reg", "loss_objectness",
        "loss_rpn_box_reg",
    ]

    training_stats_detailed = utils.PD_Stats(
        os.path.join(args.checkpoint_path, "train_stats_detailed.pkl"), 
        pd_train_header,
    )

    training_stats = utils.PD_Stats(
        os.path.join(args.checkpoint_path, "train_stats.pkl"), 
        ["loss"],
    )
    
    num_classes = 100
    train_dataset = LabeledDataset(root=args.data_path, split="training", transforms=get_transform(train=True))
    valid_dataset = LabeledDataset(root=args.data_path, split="validation", transforms=get_transform(train=False))
    
    if args.debug:
        index = list(range(10))
        train_dataset = torch.utils.data.Subset(train_dataset, index)
        valid_dataset = torch.utils.data.Subset(valid_dataset, index)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)
    
    model = get_model(num_classes, pretrained_hub=args.pretrained_hub)
    model.to(device)

    low_lr_param = []
    low_name = []
    high_lr_param = []
    high_name = []
    for name, param in model.named_parameters():
        if "backbone.body" in name:
            low_lr_param.append(param)
            low_name.append(name)
        else:
            high_lr_param.append(param)
            high_name.append(name)

    optimizer = torch.optim.SGD(
         [
            {"params": high_lr_param},
            {"params": low_lr_param, "lr": 0.0001} 
         ],
         lr=0.005,
         momentum=0.9,
         weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    if args.mode == 'resume':
        utils.restart_from_checkpoint(
            os.path.join(args.checkpoint_path, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
    start_epoch = to_restore["epoch"]

    eval_result = {}

    if (args.mode == "train") or (args.mode == "resume"):
        for epoch in range(start_epoch, args.epochs):
            # train for one epoch, printing every 10 iterations
            train_log, smooth_loss_hist = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1000)
            
            training_stats_detailed.update(train_log)
            training_stats.update_col(smooth_loss_hist)

            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'scheduler_state': lr_scheduler.state_dict()
            }
            torch.save(
                save_dict,
                os.path.join(args.checkpoint_path, "checkpoint.pth.tar"),
            )
            # save whole model instead of state_dict
            if (epoch % args.checkpoint_freq == 0) and ((epoch > 0) or (epoch == args.epochs - 1)):
                torch.save(model, os.path.join(args.checkpoint_path, f"model_{epoch}.pth"))

            # update the learning rate
            lr_scheduler.step()

            if (epoch % args.eval_freq == 0) and ((epoch > 0) or (epoch == args.epochs - 1)):
                # evaluate on the test dataset
                coco_res, _ = evaluate(model, valid_loader, device=device)
                eval_result[epoch] = coco_res.coco_eval
    # final eval
    # if (args.eval_freq % args.epochs != 0):
    coco_res, _ = evaluate(model, valid_loader, device=device)
    eval_result[args.epochs] = coco_res.coco_eval

    # with open(os.path.join(args.checkpoint_path, "eval_res.pickle"), "w") as outfile:
    #     pickle.dump(eval_result, outfile)

if __name__ == "__main__":
    main()