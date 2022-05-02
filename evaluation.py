import torch
import transforms as T
from engine import evaluate
from dataset import LabeledDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    index = list(range(10))

    valid_dataset = LabeledDataset(root='../../data/labeled', split="validation", transforms=get_transform(train=False))
    valid_dataset = torch.utils.data.Subset(valid_dataset, index)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = torch.load("model_final.pth")
    model.eval()
    model.to(device)

    evaluate(model, valid_loader, device=device)

if __name__ == "__main__":
    main()