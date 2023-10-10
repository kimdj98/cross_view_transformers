import torch
import torch.nn as nn

import torch.nn as nn


class LaneRoadCVT(nn.Module):
    def __init__(
        self,
        encoder,
        road_decoder,
        lane_decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()

        dim_total = 0
        dim_max = 0

        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)
    
        assert dim_max == dim_total

        self.encoder = encoder
        self.road_decoder = road_decoder
        self.lane_decoder = lane_decoder
        self.outputs = outputs

        self.road_to_logits = nn.Sequential(
            nn.Conv2d(self.road_decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max//2, 1))
        
        self.lane_to_logits = nn.Sequential(
            nn.Conv2d(self.lane_decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max//2, 1))

    def forward(self, batch):
        h = self.encoder(batch)
        h_road = self.road_decoder(h)
        h_lane = self.lane_decoder(h)

        h_road = self.road_to_logits(h_road)
        h_lane = self.lane_to_logits(h_lane)

        return {'road_bev': h_road, 'lane_bev': h_lane}