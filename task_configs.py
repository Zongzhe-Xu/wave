import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce, partial

# import data loaders, task-specific losses and metrics
# from data_loaders import load_imagenet, load_text, load_cifar, load_mnist, load_deepsea, load_darcy_flow, load_psicov, load_ecg, load_satellite, load_ninapro, load_cosmic, load_spherical, load_fsd, load_domainnet, load_pde, load_openml, load_drug, load_dfc2020
# from data_loaders import load_nucleotide_transformer,load_genomic_benchmarks, load_deepsea_full, load_hg38, load_text_large, load_text_llama, load_text_llama2,load_text_xs_pythia_1b, load_text_xs_flan_t5_small, load_text_xs_flan_t5_base, load_text_xs_flan_t5_large #
from utils import FocalLoss, LpLoss, conv_init, get_params_to_update, set_param_grad, set_grad_state

from torchmetrics.classification import MultilabelAccuracy
from data_loaders import load_bigearthnet, load_brick_kiln, load_EuroSAT, load_so2sat, load_cifar
from bigearth_loader import load_BigEarthNet 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################### metric and loss #################################################
def mean_iou(y_pred, y_true, num_classes=8):
    """
    Calculate mean Intersection over Union (mIoU) for semantic segmentation.
    
    Args:
    - y_true (torch.Tensor): Ground truth segmentation masks. Shape: (batch_size, num_classes, height, width)
    - y_pred (torch.Tensor): Predicted segmentation masks. Shape: (batch_size, height, width)
    - num_classes (int): Number of classes
    
    Returns:
    - miou (float): Mean Intersection over Union
    """

    y_pred = y_pred.argmax(dim=1)
    
    ious = []
    
    for cls in range(num_classes):
        true_cls = (y_true == cls)
        pred_cls = (y_pred == cls)
        
        intersection = torch.logical_and(true_cls, pred_cls).sum()
        union = torch.logical_or(true_cls, pred_cls).sum()
        
        iou = (intersection + 1e-6) / (union + 1e-6)  # Add a small value to avoid division by zero
        ious.append(iou)
    
    miou = sum(ious) / num_classes
    return miou

class inverse_score(object):
    def __init__(self, score_func):
        self.score_func = score_func

    def __call__(self, output, target):
        return 1 - self.score_func(output, target)
    
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
    score = average_precision_score(target.cpu(), torch.sigmoid(preds).detach().cpu(), average='micro') * 100.0
    return score
# def mean_average_precision(preds, target):
#     # Compute the average precision for each class
#     # pred: (batch_size, numclasses)
#     # target: (batch_size, numclasses)
#     # return: mAP
#     preds = preds.detach().cpu()
#     target = target.detach().cpu()
#     num_classes = target.shape[1]
#     aps = []
#     for i in range(num_classes):
#         ap = average_precision_score(target[:, i], preds[:, i])
#         aps.append(ap)
#     aps = torch.tensor(aps)
#     return aps.mean()

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


###################################### metric and loss #################################################

def get_data(root, dataset, batch_size, valid_split, maxsize=None):
    data_kwargs = None

    if dataset == "your_new_task": # modify this to experiment with a new task
        train_loader, val_loader, test_loader = None, None, None
    elif dataset == "eurosat":
        train_loader, test_loader = load_EuroSAT(batch_size=batch_size)
        val_loader = None
    elif dataset in ["big_earth_net"]:
        train_loader, test_loader = load_bigearthnet(batch_size=batch_size)
        val_loader = None
    elif dataset == 'brick_kiln':
        train_loader, test_loader = load_brick_kiln(batch_size=batch_size)
        val_loader = None
    elif dataset == 'BigEarth':
        train_loader, test_loader = load_BigEarthNet(batch_size=batch_size)
        val_loader = None
    elif dataset == "CIFAR10":
        train_loader, test_loader = load_cifar(root, 10, batch_size, valid_split=valid_split, maxsize=maxsize)
        val_loader = None
    # print(dataset)

    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    if not valid_split:
        val_loader = test_loader
        n_val = n_test

    return train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs


def get_config(root, args):
    dataset = args.dataset
    args.infer_label = False
    args.activation = None
    args.target_seq_len = 512 if not hasattr(args, 'target_seq_len') else args.target_seq_len
    # print("target_seq_len", args.target_seq_len)

    if dataset == "eurosat":
        dims, sample_shape, num_classes, wavelength = 2, (1, 13, 64, 64), 10, [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]
        loss = nn.CrossEntropyLoss()
    
    elif dataset == "big_earth_net":
        posweight = torch.tensor([9.8225, 45.6200, 18.4553, 36.4532, 28.9850,  2.4530, 87.4956, 36.9507,
         3.1263,  3.1876, 42.1965, 20.8579,  3.7393, 47.1928, 42.3839, 20.4592,
        34.5872, 11.5865, 23.6609, 42.0108,  2.8001, 22.5294,  2.6941, 21.3464,
        18.6271,  1.9727, 13.9365,  3.7048, 19.1816, 12.2275, 70.9424, 23.8756,
        23.7831, 87.1057, 29.9598, 15.6806,  9.4932, 39.0802, 18.2678,  2.4252,
        19.3666, 10.1545, 16.2861]).cuda()
        loss = BCEWithLogitsLoss(pos_weight=posweight, label_smoothing=0.1)
        dims, sample_shape, num_classes, wavelength = 2, (1, 12, 120, 120), 43, [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]

    elif dataset == "DFC2020":
        loss = nn.CrossEntropyLoss()
        dims, sample_shape, num_classes = 2, (1, 14, 224, 224), (1,8,96,96)
        
    elif dataset == 'brick_kiln':
        loss = nn.CrossEntropyLoss()
        dims, sample_shape, num_classes, wavelength= 2, (1, 13, 96, 96), 2, [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]

    elif dataset == 'BigEarth':
        # posweight = torch.tensor([6.4300, 45.4816, 1.8763, 8.7540, 4.7109, 4.4508, 3.0148,
        # 18.1973, 2.9350, 1.7611, 1.7313, 42.4056, 33.0567, 2.3729,
        # 425.7350, 19.4616, 333.1415, 5.3012, 6.1223]).cuda()
        # loss = BCEWithLogitsLoss(pos_weight=posweight, label_smoothing=0.1)
        loss = nn.MultiLabelSoftMarginLoss()
        dims, sample_shape, num_classes, wavelength = 2, (1, 12, 120, 120), 19, [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]
    
    elif dataset == 'forestnet':
        loss = nn.CrossEntropyLoss()
        dims, sample_shape, num_classes, wavelength = 2, (1, 6, 332, 332), 12, [0.665, 0.560, 0.490, 0.842, 1.610, 2.190]


    return dims, sample_shape, num_classes, loss, args, wavelength


def get_metric(dataset):
    if dataset == "your_new_task": # modify this to experiment with a new task
        return inverse_score(accuracy), np.min
    if dataset == 'big_earth_net':
        #metric = MultilabelAccuracy(num_labels=43, average='micro').to('cuda')
        metric = mean_average_precision
        return inverse_score(metric), np.min
    if dataset == 'DFC2020':
        metric = partial(mean_iou, num_classes=8)
        return inverse_score(metric), np.min
        # return inverse_score(accuracy), np.min
    if dataset == 'brick_kiln':
        metric = accuracy
        return inverse_score(metric), np.min
    if dataset == 'eurosat':
        metric = accuracy
        return inverse_score(metric), np.min
    if dataset == 'BigEarth':
        metric = mean_average_precision
        return inverse_score(metric), np.min


def get_optimizer(name, params):
    if name == 'SGD':
        return partial(torch.optim.SGD, lr=params['lr'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif name == 'Adam':
        return partial(torch.optim.Adam, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])
    elif name == 'AdamW':
        return partial(torch.optim.AdamW, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])


def get_scheduler(name, params, epochs=200, n_train=None):
    if name == 'StepLR':
        sched = params['sched']

        def scheduler(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(params['base'], optim_factor)  

        lr_sched_iter = False

    elif name == 'WarmupLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return f  

    elif name == 'ExpLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return params['base'] * f  

    elif name == 'SinLR':

        cycles = 0.5
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            # progress after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1. + math.cos(math.pi * float(cycles) * 2.0 * progress)))

    return scheduler, lr_sched_iter


def get_optimizer_scheduler(args, model, module=None, n_train=1):
    if module is None:
        set_grad_state(model, True, args.finetune_method)
        set_param_grad(model, args.finetune_method)
        optimizer = get_optimizer(args.optimizer.name, args.optimizer.params)(get_params_to_update(model, ""))
        lr_lambda, args.lr_sched_iter = get_scheduler(args.scheduler.name, args.scheduler.params, args.epochs, n_train)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return args, model, optimizer, scheduler

    elif module == 'embedder-without-linear':  # exclude linear
        embedder_optimizer_params = copy.deepcopy(args.optimizer.params)
        if embedder_optimizer_params['lr'] <= 0.001:
            embedder_optimizer_params['lr'] *= 10
        params_to_update = get_params_to_update(model, "", module='embedder-without-linear') #
        embedder_optimizer = get_optimizer(args.optimizer.name, embedder_optimizer_params)(params_to_update) # 
        lr_lambda, _ = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.embedder_epochs, 1)
        embedder_scheduler = torch.optim.lr_scheduler.LambdaLR(embedder_optimizer, lr_lambda=lr_lambda)

        return args, model, embedder_optimizer, embedder_scheduler

    elif module == 'embedder-with-linear': # include linear
        embedder_optimizer_params = copy.deepcopy(args.optimizer.params)
        if embedder_optimizer_params['lr'] <= 0.001:
            embedder_optimizer_params['lr'] = 0.003
        # embedder_optimizer_params['lr'] = 0.0005
        print('embedder optimizer',embedder_optimizer_params)
        embedder_optimizer = get_optimizer("SGD", embedder_optimizer_params)(get_params_to_update(model, ""))
        lr_lambda, _ = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.embedder_epochs, 1)
        embedder_scheduler = torch.optim.lr_scheduler.LambdaLR(embedder_optimizer, lr_lambda=lr_lambda)

        return args, model, embedder_optimizer, embedder_scheduler