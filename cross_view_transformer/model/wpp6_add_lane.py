import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.nn.parameter import Parameter

import numpy as np
import math 

from einops import rearrange, repeat

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

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
        backbone,
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
        
        self.height = height
        self.width = width
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.num_layers = num_layers
        dim = heads * dim_head
        self.dim = dim

        self.cvt_lane_encoder = cvt_lane_encoder
        self.cvt_road_encoder = cvt_road_encoder
        self.cvt_vehicle_encoder = cvt_vehicle_encoder
        
        self.backbone = backbone # efficientnet.EfficientNetExtractor
        
        self.modes = modes
        self.LSTM = nn.LSTM(7, dim, num_layers, batch_first=True)

        # pos_encoding = torch.Tensor(getPositionEncoding(height*width + 2*num_layers + 6*feature_height*feature_width, dim))
        # torch.manual_seed(2023)
        pos_encoding = positionalencoding2d(4, 200, 200)
        # self.register_buffer('pos_encoding', pos_encoding, persistent=True)

        self.pos_encoding = Parameter(pos_encoding)

        sa = SelfAttention(dim=dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias)
        sa_layers = [sa] * num_SAs

        # self.cross_attention = CrossAttention(dim=dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias)
        SA = SelfAttention(dim=dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias)
        SAs = [SA] * num_SAs
        self.SA_layers = nn.Sequential(*SAs)

        # extract road and vehicle features
        self.Conv = nn.Sequential(
                                nn.Conv2d(4, 32, 3, stride=1, padding=1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(),
                                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),

                                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),

                                nn.Conv2d(128, dim, 3, stride=1, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(),
                                nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),
                                
                                nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(),
                                nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # (B, dim, h, w) = (B, dim, 12, 12)
        )


        self.Img_fc = nn.Sequential(
            nn.Linear(6*feature_height*feature_width*dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
        )

        self.Feature_fc = nn.Sequential(
            nn.Linear((height*width + 2*num_layers)*dim + 35, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 24)
        )

        self.img_encoder = ImgEncoder(backbone, dim, feature_height, feature_width)

    def forward(self, batch):
        B, _, _, _, _ = batch["image"].shape

        # Encode road and vehicle features
        enc_lane = self.cvt_lane_encoder(batch)                                                                 # (b, 128, 25, 25)
        enc_road = self.cvt_road_encoder(batch)                                                                 # (b, 128, 25, 25)
        enc_vehicle = self.cvt_vehicle_encoder(batch)                                                           # (b, 128, 25, 25)

        enc_img = self.img_encoder(batch)                                                                       # (b, n, d, fh,fw)
        enc_img = rearrange(enc_img, 'b n d h w -> b (n h w) d')                                                # (b, n*fh*fw, d)

        lane_bev = enc_lane['bev']
        road_bev = enc_road['bev']
        vehicle_bev = enc_vehicle['bev']
        vehicle_center = enc_vehicle['center']

        x = torch.concat((lane_bev.detach(), road_bev.detach(), vehicle_bev.detach(), vehicle_center.detach()), dim=1)             # (B, 4, 200, 200)
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
        y = torch.concat((enc_img, x, h, c), dim=1)                                                             # (B, h*w+num_layers+num_layers, d)

        y = self.SA_layers(y)                                                                                   # (B, h*w+num_layers+num_layers, d)

        y = rearrange(y, 'B n d -> B (n d)')                                                                    # (B, (h*w+num_layers+num_layers)*d)
        state = rearrange(state, 'b t s -> b (t s)')                                                            # (B, 35)
        y = torch.concat((y, state), dim=1)                                                                     # (B, (h*w+num_layers+num_layers)*d + 35)

        y1 = y[:, :self.feature_height*self.feature_width*self.dim*6]                                            # (B, fh*fw*d)
        y2 = y[:, self.feature_height*self.feature_width*self.dim*6:]                                            # (B, (h*w+num_layers+num_layers)*d + 35)
        y1 = self.Img_fc(y1)
        y2 = self.Feature_fc(y2)
        y = torch.concat((y1, y2), dim=1)
        y = self.decoder(y)

        y = y.view(B, 12, 2)                                                                                    # (B, 12, 2)
        return y
    

def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class ImgEncoder(nn.Module):
    def __init__(self, 
                 backbone, 
                 dim:int,
                 feat_height:int,
                 feat_width:int,):
        super().__init__()
        self.backbone = backbone
        self.norm = Normalize()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= feat_width
        image_plane[:, :, 1] *= feat_height

        self.register_buffer('image_plane', image_plane, persistent=False)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape                                    # b n c h w

        image = batch['image'].flatten(0, 1)                                    # (bn) c h w
        I_inv = batch['intrinsics'].inverse()                                   # b n 3 3
        E_inv = batch['extrinsics'].inverse()                                   # b n 4 4

        features = self.backbone(self.norm(image))                              # (bn) d h w
        # features[0].shape: torch.Size([24, 32, 56, 120])
        # features[1].shape: torch.Size([24, 112, 14, 30])

        feature = features[-1]
        feature = rearrange(feature, '(b n) d h w -> b n d h w', b=b, n=n)      # b n d h w

        b, n, _, _, _ = feature.shape                                           # b n d h w

        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # b n 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
        d = E_inv @ cam                                                         # b n 4 (h w)   
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w    
        d_embed = self.img_embed(d_flat)                                        # (b n) d h w

        img_embed = d_embed - c_embed                                           # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        img_embed = rearrange(img_embed, '(b n) d h w -> b n d h w', b=b, n=n)   # b n d h w

        return img_embed                                                        
        

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