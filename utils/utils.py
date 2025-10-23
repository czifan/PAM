import logging
from functools import wraps
import torch.optim as optim
import torch 
import numpy as np 
import random
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 
import copy
import torch.nn.functional as F
from collections import defaultdict
import os 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False 

def build_logger(log_file):
    logger = logging.getLogger('SPEC')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def release_logger(logger):
    for handler in logger.handlers[:]:  # 复制 handlers 列表进行迭代
        handler.close()
        logger.removeHandler(handler)
        
def log_parser_args(args, printer):
    args_dict = vars(args)
    for key, value in args_dict.items():
        printer(f"{key}: {value}")

def update_records(records, mydict):
    for key, value in mydict.items():
        if key in records: records[key].append(value)
        else: records[key] = [value]
    return records

def build_optimizer(model, args):
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    return optimizer

def build_lr_scheduler(optimizer, args):
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min, verbose=True)
    elif args.lr_scheduler == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, verbose=True)
    return lr_scheduler

def dice_coefficient(output, target, smooth=1e-6):
    intersection = (output.float() * target.float()).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


class EMA(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1. - decay) * model_param 
        super().__init__(model, device, ema_avg, use_buffers=True)

    def module_to_device(self, device):
        self.module = self.module.to(device)
        

def load_checkpoint(model, checkpoint, checkpoint_prefix="module.", printer=print):
    key_num = 0
    state_dict = torch.load(checkpoint)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model_state_dict = model.state_dict()
    for key, value in state_dict.items():
        tmp_key = key.replace(checkpoint_prefix, "")
        if tmp_key in model_state_dict:
            model_state_dict[tmp_key] = value 
            key_num += 1
    p = model.load_state_dict(model_state_dict)
    printer(f"Loaded checkpoint from {checkpoint} ({key_num}/{len(model_state_dict)}): {p}")
    return model
        
def count_parameters(model):
    return 1.0 * sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def compute_metric(pred_dict, batch_dict):
    metric_dict = {}
    query_ys = batch_dict["query_ys"] # (B, Nq, H, W)
    object_names = batch_dict["object_name"] # (B,)
    dice = []
    object_dices = {}
    label_size = query_ys.size()[-2:]
    predictions = pred_dict["predictions"]
    resized_predictions = []
    for prediction in predictions:
        resized_prediction = F.interpolate(prediction, size=label_size, mode="bilinear", align_corners=True)
        resized_predictions.append(resized_prediction)
    resized_predictions = torch.mean(torch.stack(resized_predictions, dim=0), dim=0) 
    resized_predictions = torch.softmax(resized_predictions, dim=1)[:, 1, ...] # (B*Nq, H, W)
    resized_predictions = resized_predictions.view(*query_ys.shape[:2], *resized_predictions.shape[-2:]) # (B, Nq, H, W)
    for b in range(query_ys.shape[0]):
        object_name = object_names[b]
        dice_score = dice_coefficient(resized_predictions[b], query_ys[b]).item() # (Nq, H, W)
        dice.append(dice_score)
        if object_name not in object_dices:
            object_dices[object_name] = []
        object_dices[object_name].append(dice_score)
    metric_dict["dice"] = np.mean(dice)
    for object_name in object_dices:
        metric_dict[object_name] = np.mean(object_dices[object_name])
    return metric_dict

def plot_training_results(focus_metric, epoch_lst, train_epoch_records, save_file, split="Train"):
    _, ax = plt.subplots(1, 1, figsize=(9, 5))
    for key in train_epoch_records:
        ax.plot(epoch_lst, train_epoch_records[key], label=f"{split} {key}")
        ax.scatter(epoch_lst, train_epoch_records[key], s=16, marker='o')
    ax.legend()
    ax.grid(True)
    plt.savefig(save_file)
    plt.close()

def save_checkpoint(epoch, model, optimizer, scheduler, save_file):
    os.makedirs(save_file.replace(os.path.basename(save_file), ""), exist_ok=True)
    torch.save({
        "epoch": epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_file)