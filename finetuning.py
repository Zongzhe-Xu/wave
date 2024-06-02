import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
import copy
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling
from torchmetrics.classification import MultilabelAccuracy
from fno import Net2d
import wandb
import copy
from functools import partial
from pathlib import Path
import tifffile
from data_loaders import load_EuroSAT, load_bigearthnet, load_brick_kiln, load_so2sat, load_forestnet, load_pv4ger
# from resnet50 import ResNet50
from bigearth_loader import load_BigEarthNet
from resnet import resnet32


class BCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, label_smoothing=0.0, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = target * positive_smoothed_labels + \
                (1 - target) * negative_smoothed_labels

        loss = self.bce_with_logits(input, target)
        return loss

def count_parameters_requires_grad(model: nn.Module) -> int:
    """
    Count the total number of parameters that require gradients in a PyTorch nn.Module model.

    Args:
    model (nn.Module): The PyTorch model.

    Returns:
    int: The total number of parameters that require gradients.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class adaptive_pooler(torch.nn.Module):
    def __init__(self, out_channel=1, output_shape=None, dense=False):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool1d(out_channel)
        self.out_channel = out_channel
        self.output_shape = output_shape
        self.dense = dense

    def forward(self, x):
        if len(x.shape) == 3:
            if self.out_channel == 1 and not self.dense:
                x = x.transpose(1, 2)
            pooled_output = self.pooler(x)
            if self.output_shape is not None:
                pooled_output = pooled_output.reshape(x.shape[0], *self.output_shape)
            else:
                pooled_output = pooled_output.reshape(x.shape[0], -1)
            
        else:
            b, c, h, w = x.shape
            x = x.reshape(b, c, -1)
            pooled_output = self.pooler(x.transpose(1, 2))
            pooled_output = pooled_output.transpose(1, 2).reshape(b, self.out_channel, h, w)
            if self.out_channel == 1:
                pooled_output = pooled_output.reshape(b, h, w)

        return pooled_output
    

import torchvision.transforms as transforms
class Embeddings2D(nn.Module):

    def __init__(self, input_shape, output_shape=None, wavelen=None):
        super().__init__()
        self.wavelen = torch.tensor(wavelen).reshape(input_shape[1], 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to('cuda')
        self.embedder_type = "resnet"
        self.dense = isinstance(output_shape, tuple)
        if self.dense:
            output_shape = output_shape[1]
        if self.embedder_type == 'resnet':
            self.dash = resnet32(in_channel = 2, num_classes = 16, remain_shape = True)
            self.fusion = nn.Conv2d(16*input_shape[1], 128, kernel_size=3, padding=1)
            self.final = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1),
                nn.Conv2d(256, output_shape, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
            )
            
        elif self.embedder_type == 'fno':
            self.fno = Net2d(modes=10, width=64, input_channels= 2, output_channels = 64) # input channel is 3: (a(x, y), x, y)
            self.fusion = nn.Conv2d(64*input_shape[1], 64, kernel_size=3, padding=1)
            self.final = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1),
                nn.Conv2d(256, output_shape, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
            )


        self.norm1 = nn.BatchNorm2d(128)



    def forward(self, x, *args, **kwargs):
        #x = self.resize(x)
        b, c, height, width = x.shape
        wave = self.wavelen.repeat(b, 1, 1, height, width) # (b, c, 1, h, w)
        x = x.unsqueeze(2)
        x = torch.cat([x, wave], 2) # (b, c, 2, h, w)
        x = x.flatten(start_dim=0, end_dim=1) # (b*c, 2, h, w)
        #x = self.maybe_pad(x, height, width)
        if self.embedder_type == 'resnet':
            x = self.dash(x) #(b*c, 16, h, w)
            x = x.reshape(b, c, 16, height, width) #(b, c, 16, h, w)
            x = x.flatten(start_dim=1, end_dim=2) #(b, c*16, h, w)
            x = self.fusion(x) #(b, 64, h, w)
            x = self.norm1(x)
            if self.dense:
                xfno = self.final(x.permute(0,2,3,1)).permute(0,3,1,2)
            else:
                xfno = self.final(x).squeeze(-1).squeeze(-1)
        elif self.embedder_type == 'fno':
            x = x.permute(0,2,3,1) #(b*c, h, w, 2)
            x = self.fno(x) #(b*c, h, w, 64)
            x = x.permute(0,3,1,2) #(b*c, 64, h, w)
            x = x.reshape(b, c, 64, height, width) #(b, c, 64, h, w)
            x = x.flatten(start_dim=1, end_dim=2) #(b, c*64, h, w)
            x = self.fusion(x) #(b, 64, h, w)
            x = self.norm1(x)
            if self.dense:
                xfno = self.final(x.permute(0,2,3,1)).permute(0,3,1,2)
            else:
                xfno = self.final(x).squeeze(-1).squeeze(-1)
        return xfno


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    res = res[0] if len(res) == 1 else res
    return res


from sklearn.metrics import average_precision_score
def mean_average_precision(preds, target):
    # Compute the average precision for each class
    # pred: (batch_size, numclasses)
    # target: (batch_size, numclasses)
    # return: mAP
    preds = preds.detach().cpu()
    target = target.detach().cpu()
    num_classes = target.shape[1]
    aps = []
    for i in range(num_classes):
        ap = average_precision_score(target[:, i], preds[:, i])
        aps.append(ap)
    aps = torch.tensor(aps)
    return aps.mean()
# def mean_average_precision(preds, target):
#     score = average_precision_score(target.cpu(), torch.sigmoid(preds).detach().cpu(), average='micro') * 100.0
#     return score
            
    
def schedule(step, warmup=5, decay=70):
    if step < warmup:
        return float(step) / float(warmup)
    else:
        current = decay - step
        total = decay - warmup
        return max(current / total, 0)


    
if __name__ == '__main__':
    seed = 0
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    wb = False
    if wb:
        wandb.login(key="1ecb12fc549eac01ee610250a9f1bb724f04a3ee")
        print('wandb login')
        run = wandb.init(
            # Set the project where this run will be logged
            name = 'swin_transformer_base_resized_label', 
            project="DFC2020",
            config= {
                "learning_rate": 0.003,
                "batch_size": 16,
                "momentum": 0.9,
                "weight_decay": 0.001,
                "num_epochs": 25,
            },
            )
        print('wandb init')
    # train_loader, test_loader = load_pv4ger()
    # train_loader, test_loader = load_forestnet(batch_size=1)
    # train_loader, test_loader = load_so2sat(channels=[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,19,20])
    # train_loader, test_loader = load_EuroSAT()
    # train_loader, test_loader = load_BigEarthNet(batch_size=16, channels=[3,2,1])
    train_loader, test_loader = load_bigearthnet(batch_size=4)
    # train_loader, test_loader = load_brick_kiln(batch_size=8)
    model = Embeddings2D(input_shape=(1, 12, 120, 120), output_shape=43, wavelen=[0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190])
    # model = Embeddings2D(input_shape=(1, 6, 332, 332), output_shape=12, wavelen=[0.665, 0.560, 0.490, 0.842, 1.610, 2.190])
    # model = resnet32(in_channel = 6, num_classes = 12, remain_shape = False)
    posweight = torch.tensor([9.8225, 45.6200, 18.4553, 36.4532, 28.9850,  2.4530, 87.4956, 36.9507,
         3.1263,  3.1876, 42.1965, 20.8579,  3.7393, 47.1928, 42.3839, 20.4592,
        34.5872, 11.5865, 23.6609, 42.0108,  2.8001, 22.5294,  2.6941, 21.3464,
        18.6271,  1.9727, 13.9365,  3.7048, 19.1816, 12.2275, 70.9424, 23.8756,
        23.7831, 87.1057, 29.9598, 15.6806,  9.4932, 39.0802, 18.2678,  2.4252,
        19.3666, 10.1545, 16.2861]).cuda()
    criterion = BCEWithLogitsLoss(pos_weight=posweight, label_smoothing=0.1)
    # posweight = torch.tensor([6.4300, 45.4816, 1.8763, 8.7540, 4.7109, 4.4508, 3.0148,
    #     18.1973, 2.9350, 1.7611, 1.7313, 42.4056, 33.0567, 2.3729,
    #     425.7350, 19.4616, 333.1415, 5.3012, 6.1223]).cuda()
    # criterion = BCEWithLogitsLoss(pos_weight=posweight, label_smoothing=0.1)
    # metric = MultilabelAccuracy(num_labels=43, average='micro').to('cuda')
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = nn.MultiLabelSoftMarginLoss()
    #metric = partial(mean_iou, num_classes=8)
    # metric = accuracy
    metric = mean_average_precision
    model = model.cuda()
    # for i, (X, y) in enumerate(train_loader):
    #     X = X.cuda()
    #     y = y.cuda()
    #     print(X.shape)
    #     y_pred = model(X)
    #     print(y_pred.shape)
    #     print(y.shape)
    #     score = metric(y_pred, y)
    #     loss = criterion(y_pred, y)
    #     break
    # exit()
    accumulations = 1000
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: schedule(step))
    best_acc = 0
    model.train()
    for epoch in range(60):
        print("Epoch: %d" % (epoch))
        train_loss = 0
        train_score = 0
        test_loss = 0
        test_score = 0
        num_batch = 0
        num_samples = 0
        y_pred_all = []
        y_true_all = []
        total_samples = 0
        
        for i, (X, y) in enumerate(train_loader):
            X = X.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            y_pred = model(X)
            if i == 0:
                print(y_pred.shape)
                print(y.shape)
            loss = criterion(y_pred, y)
            y_pred_all.append(y_pred)
            y_true_all.append(y)
            num_samples += y.shape[0]
            #score = metric(y_pred.detach(), y.detach())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            #train_score += score.item()
            num_batch += 1
            if num_samples >= accumulations or i == len(train_loader) - 1:
            # if i == len(train_loader) - 1:
                y_pred_all = torch.cat(y_pred_all, dim=0)
                y_true_all = torch.cat(y_true_all, dim=0)
                score = metric(y_pred_all, y_true_all)
                total_samples += 1
                train_score += score.item()
                num_samples = 0
                y_pred_all = []
                y_true_all = []

        scheduler.step()
        train_loss /= num_batch
        train_score /= total_samples
        num_batch = 0
        num_samples = 0
        y_pred_all = []
        y_true_all = []
        total_samples = 0
        with torch.no_grad():
            model.eval()
            for i, (X, y) in enumerate(test_loader):
                X = X.cuda()
                y = y.cuda()
                y_pred = model(X)
                loss = criterion(y_pred, y)
                y_pred_all.append(y_pred)
                y_true_all.append(y)
                num_samples += y.shape[0]
                #score = metric(y_pred.detach(), y.detach())
                test_loss += loss.item()
                #test_score += score.item()
                num_batch += 1
                if num_samples >= accumulations or i == len(test_loader) - 1:
                # if i == len(test_loader) - 1:
                    y_pred_all = torch.cat(y_pred_all, dim=0)
                    y_true_all = torch.cat(y_true_all, dim=0)
                    score = metric(y_pred_all, y_true_all)
                    total_samples += 1
                    test_score += score.item()
                    num_samples = 0
                    y_pred_all = []
                    y_true_all = []
            test_loss /= num_batch
            test_score /= total_samples
            if test_score > best_acc:
                best_acc = test_score
                #torch.save(model.state_dict(), 'results/cnn_embedder.pt')
            print("Epoch: %d, lr: %f, Train Loss: %f, Train Acc: %f, Test Loss: %f, Test Acc: %f" % (epoch, optimizer.param_groups[0]['lr'], train_loss, train_score, test_loss, test_score))
            if wb:
                wandb.log({
                        'train_loss': train_loss,  
                        'train_acc': train_score,
                        'test_loss': test_loss,
                        'test_acc': test_score,
                        'lr': optimizer.param_groups[0]['lr']
                    })
    print("Best Acc: %f" % (best_acc))