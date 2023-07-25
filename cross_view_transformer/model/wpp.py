import torch.nn as nn
import torch

class WppNetwork(nn.Module):
    def __init__(
        self,
        cvt_road_encoder,
        cvt_vehicle_encoder,
        modes: int = 10,
        outputs: dict = {'bev': [0, 1]},
    ):
        
        super().__init__()
        
        # TODO: transfer weights to encoders
        #       scripts/train.py

        self.cvt_road_encoder = cvt_road_encoder
        self.cvt_vehicle_encoder = cvt_vehicle_encoder

        # TODO: move to loss calculation
        # self.road_data = None
        # self.vehicle_data = None
        # self.road_data_label_indices = None
        # self.vehicle_data_label_indices = None

        self.state_projection = nn.Sequential(nn.Linear(7, 128))
        
        self.fc = nn.Sequential(
                                nn.Linear(9251, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, modes*(12*2 + 1)) # predict 10 trajectories for 6 seconds with probabilities
        )

        self.conv = nn.Sequential(
                                nn.Conv2d(3, 32, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),

                                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),

                                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),

                                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),

                                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),

                                nn.Flatten()
        )

    def forward(self, batch):
        B, _, _, _, _ = batch["image"].shape

        # Encode road and vehicle features
        enc_road = self.cvt_road_encoder(batch)                                                                          # (B, 128, 25, 25)
        enc_vehicle = self.cvt_vehicle_encoder(batch)                                                                    # (B, 128, 25, 25)

        road_bev = enc_road['bev']
        vehicle_bev = enc_vehicle['bev']
        vehicle_center = enc_vehicle['center']

        x = torch.concat((road_bev.detach(), vehicle_bev.detach(), vehicle_center.detach()), dim=1) # (B, 3, 200, 200)

        x = self.conv(x)

        # Gather state data
        state_components = [batch["past_coordinate"], batch["past_vel"], 
                            batch["past_acc"], batch["past_yaw"]]
        
        state = torch.cat(state_components, dim=2)    # (B, 5, 7)

        # Reshape encoded features and state data for output
        reshaped_components = [comp.reshape(B, -1) for comp in [x, state]]

        # Concatenate reshaped data along dimension 1
        output = torch.cat(reshaped_components, dim=1) # (B, 9251)

        # predict 10 trajectories for 6 seconds with probabilities
        output = self.fc(output) # (B, M*(12*2+1))

        output = output.view(B, -1, 12*2 + 1) # (B, M, 12*2+1)

        return output
