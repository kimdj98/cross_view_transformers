import torch
import logging
import torch.nn.functional as F
import torch.nn as nn

from fvcore.nn import sigmoid_focal_loss

logger = logging.getLogger(__name__)

class SigmoidFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=-1.0,
        gamma=2.0,
        reduction='mean'
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label):
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)


class BinarySegmentationLoss(SigmoidFocalLoss):
    def __init__(
        self,
        label_indices=None,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.label_indices = label_indices
        self.min_visibility = min_visibility

    def forward(self, pred, batch):
        if isinstance(pred, dict):
            pred = pred['bev']

        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)

        loss = super().forward(pred, label)

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()


class CenterLoss(SigmoidFocalLoss):
    def __init__(
        self,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.min_visibility = min_visibility

    def forward(self, pred, batch):
        pred = pred['center']
        label = batch['center']
        loss = super().forward(pred, label)

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()


class MultipleLoss(torch.nn.ModuleDict):
    """
    losses = MultipleLoss({'bce': torch.nn.BCEWithLogitsLoss(), 'bce_weight': 1.0})
    loss, unweighted_outputs = losses(pred, label)
    """
    def __init__(self, modules_or_weights):
        modules = dict()
        weights = dict()

        # Parse only the weights
        for key, v in modules_or_weights.items():
            if isinstance(v, float):
                weights[key.replace('_weight', '')] = v

        # Parse the loss functions
        for key, v in modules_or_weights.items():
            if not isinstance(v, float):
                modules[key] = v

                # Assign weight to 1.0 if not explicitly set.
                if key not in weights:
                    logger.warn(f'Weight for {key} was not specified.')
                    weights[key] = 1.0

        assert modules.keys() == weights.keys()

        super().__init__(modules)

        self._weights = weights

    def forward(self, pred, batch):
        # used for concurrently segmenting road and lane
        if 'focal_road' in self.keys() and 'focal_lane' in self.keys():
            lane_loss = self['focal_lane'](pred['lane_bev'], batch)
            road_loss = self['focal_road'](pred['road_bev'], batch)
            outputs = {'focal_lane': lane_loss, 'focal_road': road_loss}
            total = sum(self._weights[k] * o for k, o in outputs.items())
        else:
            outputs = {k: v(pred, batch) for k, v in self.items()}
            total = sum(self._weights[k] * o for k, o in outputs.items())

        return total, outputs

class MinMSELoss(torch.nn.Module):
    def __init__(self, modes:int, weight:float=1.0):
        super().__init__()
        self.modes = modes
        self.weight = weight

    def forward(self, pred, batch):
        waypoint_pred, _ = pred
        B, M, _, _  = waypoint_pred.shape

        label = batch["label_waypoint"].unsqueeze(1)

        # calculate angle between predicted endpoint and label endpoint
        # TODO: maybe make this a separate function
        pred_endpoint = waypoint_pred[:, :, -1].detach()                                    # (B, M, 2)
        label_endpoint = label[:, :, -1].detach()                                           # (B, 1, 2)

        unit_pred_end = pred_endpoint / pred_endpoint.norm(dim=2, keepdim=True)             # (B, M, 2)
        unit_label_end = label_endpoint / label_endpoint.norm(dim=2, keepdim=True)          # (B, 1, 2)

        label_end_norm = label_endpoint.norm(dim=2, keepdim=True)                           # (B, 1, 1)

        cos_sim = torch.sum(unit_pred_end * unit_label_end, dim=2)                          # (B, M)

        angles = torch.acos(cos_sim)                                                        # (B, M)
        angles_degrees = angles * (180 / torch.pi)                                          # (B, M)

        # find candidate where angle is less than 5 degrees 
        candidate = angles_degrees < 5                                                      # (B, M)

        # fill 1 where there is no candidate for all modes
        exist_cand = (candidate.sum(dim=1) == 0)                                            # (B)
        candidate[exist_cand] = 1
        
        # fill 1 where for stationary data
        stationary = (label_end_norm <= 1e-1).squeeze()                                    # (B)
        candidate[stationary] = 1

        # fill inf where there is no candidate
        candidate = candidate.float()
        candidate[candidate == 0] = 10000. # think of it as float('inf')
        
        SE = (waypoint_pred - label)**2
        SE[:, :, :, 0] *= self.weight
        MSE = torch.sum(SE, dim=[2,3]) / 12
        MSE *= candidate

        min_index = torch.argmin(MSE, dim=1)
        min_MSE = MSE[torch.arange(B), min_index]
        
        return min_MSE.sum()
    

def MSE(pred, label):
    if pred.dim() == label.dim() + 1:
        B, M, _, _ = pred.shape
        label = label.unsqueeze(1)
        SE = (pred - label)**2
        MSE = torch.sum(SE, dim=[2,3]) / 12

        min_index = torch.argmin(MSE, dim=1)
        min_MSE = MSE[torch.arange(B), min_index]

        # MSE = MSE # average mode
        return min_MSE.sum()
    else:
        SE = (pred - label)**2
        MSE = torch.mean(SE, dim=[1,2])
        return MSE.sum()


class MSELoss(torch.nn.Module):
    def __init__(self, modes:int, weight:float=1.0):
        super().__init__()
        self.loss_fn = MSE
        self.modes = modes
        self.weight = weight

    def forward(self, pred, batch):
        label = batch["label_waypoint"]

        if isinstance(pred, tuple):
            pred = pred[0]
            assert isinstance(pred, torch.Tensor), f'pred is not a tensor: {type(pred)}' 

        MSE = self.loss_fn(pred, label)

        return MSE
    
    
class CELoss(torch.nn.Module):
    def __init__(self, modes:int):
        super().__init__()
        self.modes = modes
        
    def forward(self, pred, batch):
        waypoint_pred, p = pred
        prob = F.softmax(p, dim=1)
        B, M = prob.shape

        label = batch["label_waypoint"].unsqueeze(1)

        pred_endpoint = waypoint_pred[:, :, -1].detach()                                    # (B, M, 2)
        label_endpoint = label[:, :, -1].detach()                                           # (B, 1, 2)

        pred_end_norm = pred_endpoint / pred_endpoint.norm(dim=2, keepdim=True)             # (B, M, 2)
        label_end_norm = label_endpoint / label_endpoint.norm(dim=2, keepdim=True)          # (B, 1, 2)

        cos_sim = torch.sum(pred_end_norm * label_end_norm, dim=2)                          # (B, M)

        angles = torch.acos(cos_sim)                                                        # (B, M)
        angles_degrees = angles * (180 / torch.pi)                                          # (B, M)

        # find candidate where angle is less than 5 degrees 
        candidate = angles_degrees < 5

        # fill 1 where there is no candidate for all modes
        exist_cand = (candidate.sum(dim=1) == 0)

        # fill inf where there is no candidate
        candidate[exist_cand] = 1
        candidate = candidate.float()
        candidate[candidate == 0] = float('inf')
        
        SE = (waypoint_pred - label)**2
        SSE = torch.sum(SE, dim=[2,3])

        SSE *= candidate

        min_index = torch.argmin(SSE, dim=1)
        min_index = F.one_hot(min_index, num_classes=self.modes).float()

        CELoss = torch.sum(-min_index * torch.log(prob + 1e-8), dim=1)
        return CELoss.sum()


def weighted_MSE(pred, label, weight:float):
    SE = (pred - label)**2
    SE[:,:,0] *= weight

    return torch.sum(SE, dim=[1,2]) / 12 # average 12 time steps


class weighted_MSELoss(torch.nn.Module):
    def __init__(self, weight:float):
        super().__init__()
        self.loss_fn = weighted_MSE
        self.weight = weight

    def forward(self, pred, batch):
        states = ["label_waypoint", "label_vel", "label_acc", "label_yaw"]
        label = batch[states[0]]
        
        MSE = weighted_MSE(pred, label, self.weight)
        
        return MSE.sum()


class velocity_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, pred, batch):
        return self.loss_fn(pred, batch["velocity_map"].float())