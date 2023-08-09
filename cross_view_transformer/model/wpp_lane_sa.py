import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.nn.parameter import Parameter

import math
import numpy as np

from einops import rearrange, repeat

# def getPositionEncoding(seq_len, d, n=10000):
#     P = np.zeros((seq_len, d))
#     for k in range(seq_len):
#         for i in np.arange(int(d/2)):
#             denominator = np.power(n, 2*i/d)
#             P[k, 2*i] = np.sin(k/denominator)
#             P[k, 2*i+1] = np.cos(k/denominator)
#     return P

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class WppNetwork(nn.Module):
    def __init__(
        self,
        cvt_lane_encoder,
        cvt_road_encoder,
        cvt_vehicle_encoder,
        modes:int,
        num_layers:int,         # number of layers in the LSTM
        heads:int,
        dim_head:int,
        qkv_bias:bool,
        num_SAs:int,            # number of self attentions
        height:int,             # for position encoding
        width:int,              # for position encoding
        feature_height:int,
        feature_width:int,
        # device: list,
        outputs: dict = {'bev': [0, 1]},
    ):
        
        super().__init__()
        self.cvt_lane_encoder = cvt_lane_encoder
        self.cvt_road_encoder = cvt_road_encoder
        self.cvt_vehicle_encoder = cvt_vehicle_encoder

        dim = heads * dim_head
        self.dim = dim

        self.modes = modes
        self.LSTM = nn.LSTM(7, 256, num_layers, batch_first=True)

        pos_encoding = torch.Tensor(positionalencoding2d(4, 200, 200))
        self.pos_encoding = Parameter(pos_encoding)
        # self.register_buffer('pos_encoding', pos_encoding, persistent=False)

        sa = SelfAttention(dim=dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias)
        sa_layers = [sa] * num_SAs

        # self.cross_attention = CrossAttention(dim=dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias)
        SA = SelfAttention(dim=dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias)
        SAs = [SA] * num_SAs
        self.SA_layers = nn.Sequential(*SAs)

        # extract road and vehicle features
        self.Conv = nn.Sequential(
                                nn.Conv2d(4, 32, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),

                                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),

                                nn.Conv2d(128, dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),
                                
                                nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # (B, dim, h, w) = (B, 256, 12, 12)
        )

        # # split the output to query, key, value
        # self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        # self.to_k = nn.Linear(dim, dim, bias=qkv_bias)

        self.fc = nn.Sequential(
            nn.Linear((height*width + num_layers*2)*dim+35, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 24),
            )

    def forward(self, batch):
        B, _, _, _, _ = batch["image"].shape

        # Encode road and vehicle features
        enc_lane = self.cvt_lane_encoder(batch)
        enc_road = self.cvt_road_encoder(batch)                                                                 # (B, 128, 25, 25)
        enc_vehicle = self.cvt_vehicle_encoder(batch)                                                           # (B, 128, 25, 25)

        lane_bev = enc_lane['bev']
        road_bev = enc_road['bev']
        vehicle_bev = enc_vehicle['bev']
        vehicle_center = enc_vehicle['center']

        x = torch.concat((lane_bev.detach(), road_bev.detach(), vehicle_bev.detach(), vehicle_center.detach()), dim=1)             # (B, 3, 200, 200)
        x += self.pos_encoding
        x = self.Conv(x)                                                                                        # (B, d, h, w)

        B, d, _, _ = x.shape

        # Gather state data
        type = ["past_coordinate", "past_vel", "past_acc", "past_yaw"]
        state_components = [batch["past_coordinate"], batch["past_vel"], batch["past_acc"], batch["past_yaw"]]
        state = torch.cat(state_components, dim=2)                                                              # (B, 5, 7)
        
        coord_components = [batch["past_coordinate"]]
        coord = batch["past_coordinate"]                                                                        # (B, 5, 2)
        coord = coord.view(B, -1)                                                                               # (B, 10)

        # Pass state data through LSTM
        _, (h, c) = self.LSTM(state)
                                                                                                                # h: num_layer, B, d
                                                                                                                # c: num_layer, B, d

        h = rearrange(h, 'n b d -> b n d')                                                                      # (B, num_layers, d)
        c = rearrange(c, 'n b d -> b n d')                                                                      # (B, num_layers, d)
        x = rearrange(x, 'b d h w -> b (h w) d')                                                                # (B, h*w, d)
        y = torch.concat((x, h, c), dim=1)                                                                      # (B, h*w+num_layers+num_layers, d)

        y = self.SA_layers(y)                                                                                   # (B, h*w+num_layers+num_layers, d)

        y = rearrange(y, 'B n d -> B (n d)')                                                                    # (B, (h*w+num_layers+num_layers)*d)
        state = rearrange(state, 'b t s -> b (t s)')                                                            # (B, 35)
        y = torch.concat((y, state), dim=1)                                                                     # (B, (h*w+num_layers+num_layers)*d + 35)
        y = self.fc(y)


        y = y.view(B, 12, 2)                                                                                    # (B, 12, 2)
        return y


class SelfAttention(nn.Module):
    def __init__(self, 
                dim:int,
                heads:int, 
                dim_head:int, 
                qkv_bias:bool, 
                norm=nn.LayerNorm,
                ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, x):

        q = self.to_q(x)                            # q: (b, n, dim_head*heads)
        k = self.to_k(x)                            # k: (b, n, dim_head*heads)
        v = self.to_v(x)                            # v: (b, n, dim_head*heads)

        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # q: (B, l, dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # k: (B, h*w, dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # v: (B, h*w, dim_head)

        dot = self.scale * torch.einsum('B Q d, B K d -> B Q K', q, k)              # dot: (B, n, n)
        att = dot.softmax(dim=-1)                                                   # att: (B, n, n)
        z = torch.einsum('B Q K, B V d -> B Q d', att, v)                           # a: (B, n, dim_head)
        z = rearrange(z, '(b m) Q d -> b Q (m d)', m=self.heads, d=self.dim_head)   # a: (b, n, dim_head*heads)

        # if skip is not None:
        #     # skip connection (after attention)
        # pre skip connection (after attention)
        z += x

        skip = z

        z = self.prenorm(z)
        z = self.mlp(z)

        # skip connection (after mlp) (residual connection)
        z += skip

        # z = rearrange(z, 'b n d -> n b d') # (l, b, d)
        return z