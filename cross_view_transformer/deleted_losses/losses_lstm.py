import torch
import logging
import torch.nn.functional as F

from fvcore.nn import sigmoid_focal_loss

logger = logging.getLogger(__name__)


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
    

class MinMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, batch):
        B, M, _ = pred.shape

        coord = pred[:, :, :-1]
        label = batch["label_waypoint"].view(-1, 24)[:, None, :]

        SE = (coord - label)**2
        SSE = torch.sum(SE, dim=2)

        min_MSE = torch.min(SSE, dim=1)[0] / 12 # 12 time steps

        return min_MSE.sum() / B
    

def MSE(pred, label):
    SE = (pred - label)**2
    MSE = torch.mean(SE, dim=[1,2])
    return MSE.sum()


class MSELoss(torch.nn.Module):
    def __init__(self, modes:int):
        super().__init__()
        self.loss_fn = MSE
        self.modes = modes

    def forward(self, pred, batch):
        states = ["label_waypoint", "label_vel", "label_acc", "label_yaw"]

        # label = torch.concat((batch[states[0]], batch[states[1]], batch[states[2]], batch[states[3]]), dim=2)

        # label = torch.concat((batch[states[0]]), dim=2)
        label = batch[states[0]]

        MSE = self.loss_fn(label, pred)

        return MSE
    
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

        return CELoss
    