import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np
from utils import dice_coefficient

class CombinedCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.criteria = []
        self.weights = args.criterion_weight_lst

        for criterion_name in args.criterion_name_lst:
            try: self.criteria.append(eval(criterion_name)(args))
            except: raise ValueError(f"Unknown criterion: {criterion_name}")

    def forward(self, pred, batch):
        loss_dict = {"loss": 0.0}
        for weight, criterion in zip(self.weights, self.criteria):
            tmp_loss = criterion(pred, batch)
            loss_dict["loss"] += weight * tmp_loss 
            loss_dict[str(criterion)] = tmp_loss
        return loss_dict

class MultiScaleSoftDiceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, pred, batch):
        query_ys = batch["query_ys"] # (B, Nq, H, W)
        if len(query_ys.shape) == 4:
            query_ys = query_ys.view(query_ys.shape[0]*query_ys.shape[1], *query_ys.shape[2:]) # (BNq, H, W)
        label_size = query_ys.size()[-2:] # (H, W)
        loss = []
        proposing_region_predictions = pred["predictions"]
        for proposing_region_prediction in proposing_region_predictions:
            resized_proposing_region_prediction = F.interpolate(proposing_region_prediction, size=label_size, mode="bilinear", align_corners=True)
            resized_proposing_region_prediction = torch.softmax(resized_proposing_region_prediction, dim=1)[:, 1, ...]
            loss.append(1.0 - dice_coefficient(resized_proposing_region_prediction, query_ys))
        loss = torch.mean(torch.stack(loss, dim=0), dim=0)
        return loss

    def __str__(self):
        return "msdiceloss"
    
class MultiScaleCELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, pred, batch):
        query_ys = batch["query_ys"] # (B, Nq, H, W)
        if len(query_ys.shape) == 4:
            query_ys = query_ys.view(query_ys.shape[0]*query_ys.shape[1], *query_ys.shape[2:]).long() # (BNq, H, W)
        label_size = query_ys.size()[-2:] # (H, W)
        loss = []
        proposing_region_predictions = pred["predictions"]
        for proposing_region_prediction in proposing_region_predictions:
            resized_proposing_region_prediction = F.interpolate(proposing_region_prediction, size=label_size, mode="bilinear", align_corners=True)
            loss.append(self.ce_criterion(resized_proposing_region_prediction, query_ys))
        loss = torch.mean(torch.stack(loss, dim=0), dim=0)
        return loss

    def __str__(self):
        return "msceloss"


class MultiScaleFocalCELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def focal_ce_criterion(self, inputs, proposing_preds, targets):
        proposing_preds = torch.argmax(proposing_preds, dim=1)
        targets[proposing_preds == targets] = -1
        return self.ce_criterion(inputs, targets.long())

    def forward(self, pred, batch):
        query_ys = batch["query_ys"] # (B, Nq, H, W)
        if len(query_ys.shape) == 4:
            query_ys = query_ys.view(query_ys.shape[0]*query_ys.shape[1], *query_ys.shape[2:]).long() # (BNq, H, W)
        label_size = query_ys.size()[-2:] # (H, W)
        loss = []
        proposing_region_predictions = pred["proposing_region_predictions"]
        predictions = pred["predictions"]
        for proposing_region_prediction, prediction in zip(predictions, proposing_region_predictions):
            resized_prediction = F.interpolate(prediction, size=label_size, mode="bilinear", align_corners=True)
            resized_proposing_region_prediction = F.interpolate(proposing_region_prediction, size=label_size, mode="bilinear", align_corners=True)
            loss.append(self.focal_ce_criterion(resized_prediction, resized_proposing_region_prediction, query_ys))
        loss = torch.mean(torch.stack(loss, dim=0), dim=0)
        return loss

    def __str__(self):
        return "msfocalceloss"
