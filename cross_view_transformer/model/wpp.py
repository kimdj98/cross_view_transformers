import torch.nn as nn


class WppNetwork(nn.Module):
    def __init__(
        self,
        cvt_road_encoder,
        cvt_vehicle_encoder,
    ):
        super().__init__()
        
        self.cvt_road_encoder = cvt_road_encoder
        self.cvt_vehicle_encoder = cvt_vehicle_encoder
        

    def forward(self, batch):

        enc_road = self.cvt_road_encoder(batch)
        enc_vehicle = self.cvt_vehicle_encoder(batch)

        # TODO: return concatenated encoder outputs
        return None
