import torch
import logging
import torch.nn.functional as F

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
        outputs = {k: v(pred, batch) for k, v in self.items()}
        total = sum(self._weights[k] * o for k, o in outputs.items())

        return total, outputs

# # loss for wpp
# class WppLoss(torch.nn.Module):
#     def __init__(self, alpha=1.0, beta=1.0):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta

#     def forward(self, pred, batch):
#         B, M, _ = pred.shape

#         logit = pred[:, :,  -1]
#         coord = pred[:, :, :-1]

#         label = batch["label_waypoint"].view(-1, 24)[:, None, :]

#         softmax = torch.nn.Softmax(dim=1)
#         prob = softmax(logit)

#         SE = (coord - label)**2
#         SSE = torch.sum(SE, dim=2)

#         min_MSE = torch.min(SSE, dim=1)[0] / M
#         argmin_SSE = torch.argmin(SSE, dim=1)

#         CELoss = F.cross_entropy(logit, argmin_SSE)

#         return min_MSE.sum() + CELoss.sum()
    

class MinMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, batch):
        B, M, _ = pred.shape

        coord = pred[:, :, :-1]
        label = batch["label_waypoint"].view(-1, 24)[:, None, :]

        SE = (coord - label)**2
        SSE = torch.sum(SE, dim=2)

        min_MSE = torch.min(SSE, dim=1)[0] / M 

        return min_MSE.sum() / B
    
class MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, batch):
        B, M, _ = pred.shape

        coord = pred[:, :, :-1]
        label = batch["label_waypoint"].view(-1, 24)[:, None, :]

        SE = (coord - label)**2
        SSE = torch.sum(SE, dim=2)

        min_MSE = torch.sum(SSE, dim=1) / M 

        return min_MSE.sum() / B
    
class CELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, batch):
        B, M, _ = pred.shape

        logit = pred[:, :,  -1]
        coord = pred[:, :, :-1]

        label = batch["label_waypoint"].view(-1, 24)[:, None, :]

        softmax = torch.nn.Softmax(dim=1)

        SE = (coord - label)**2
        SSE = torch.sum(SE, dim=2)

        argmin_SSE = torch.argmin(SSE, dim=1)

        CELoss = F.cross_entropy(logit, argmin_SSE)

        return CELoss.sum() / B
    