import torch

from torchmetrics import Metric
from typing import Any, Callable, List, Optional


class BaseIoUMetric(Metric):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, thresholds=[0.4, 0.5]):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        thresholds = torch.FloatTensor(thresholds)

        self.add_state('thresholds', default=thresholds, dist_reduce_fx='mean')
        self.add_state('tp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')

    def update(self, pred, label):
        pred = pred.detach().sigmoid().reshape(-1)
        label = label.detach().bool().reshape(-1)

        pred = pred[:, None] >= self.thresholds[None]
        label = label[:, None]

        self.tp += (pred & label).sum(0)
        self.fp += (pred & ~label).sum(0)
        self.fn += (~pred & label).sum(0)

    def compute(self):
        thresholds = self.thresholds.squeeze(0)
        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)

        return {f'@{t.item():.2f}': i.item() for t, i in zip(thresholds, ious)}


class IoUMetric(BaseIoUMetric):
    def __init__(self, label_indices: List[List[int]], min_visibility: Optional[int] = None):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples

        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__()

        self.label_indices = label_indices
        self.min_visibility = min_visibility

    def update(self, pred, batch):
        if isinstance(pred, dict):
            pred = pred['bev']                                                              # b c h w

        label = batch['bev']                                                                # b n h w
        label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
        label = torch.cat(label, 1)                                                         # b c h w

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            mask = mask[:, None].expand_as(pred)                                            # b c h w

            pred = pred[mask]                                                               # m
            label = label[mask]                                                             # m

        return super().update(pred, label)

class ADEMetric(Metric):
    def __init__(self, modes:int,):
        super().__init__()
        self.min_ADE = 0.0
        self.total = 0
        self.modes = modes
        self.MSE = torch.nn.MSELoss(reduction='mean')
        self.SSE = torch.nn.MSELoss(reduction='sum')

    def update(self, pred, batch):
        B, _, _ = pred.shape

        coord = pred.detach()
        label = batch['label_waypoint'].detach()

        SE = (coord - label)**2
        MSE = torch.sum(torch.sqrt(torch.sum(SE, dim=2)), dim=1) / 12
        
        self.min_ADE += torch.sum(MSE)
        self.total += B

    def compute(self):
        result = self.min_ADE / self.total
        self.reset()
        return result
    
    def reset(self):
        self.min_ADE = 0.0
        self.total = 0

class FDEMetric(Metric):
    def __init__(self, modes:int,):
        super().__init__()
        self.min_FDE = 0.0
        self.total = 0
        self.modes = modes
        self.SSE = torch.nn.MSELoss(reduction='sum')

    def update(self, pred, batch):
        B, _, _ = pred.shape

        coord = pred.detach()
        label = batch['label_waypoint'].detach()

        coord = coord[:, -2:, :]
        label = label[:, -2:, :]

        SE = (coord - label)**2
        MSE = torch.sum(torch.sqrt(torch.sum(SE, dim=2)), dim=1)

        self.min_FDE += torch.sum(MSE)
        self.total += B

    def compute(self):
        result = self.min_ADE / self.total
        self.reset()
        return result
    
    def reset(self):
        self.min_ADE = 0.0
        self.total = 0
