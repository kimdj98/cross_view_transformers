import torch.nn as nn
import torch

from einops import rearrange, repeat

class WppNetwork(nn.Module):
    def __init__(
        self,
        cvt_road_encoder,
        cvt_vehicle_encoder,
        modes:int,
        dim:int,
        heads:int,
        dim_head:int,
        qkv_bias:bool,
        num_layers:int,
        outputs: dict = {'bev': [0, 1]},
    ):
        
        super().__init__()
        
        self.cvt_road_encoder = cvt_road_encoder
        self.cvt_vehicle_encoder = cvt_vehicle_encoder

        self.modes = modes
        self.RNNs = nn.ModuleList([nn.RNN(2, 256, 2, batch_first=True)] * modes)
        # self.LSTMs = nn.ModuleList([nn.LSTM(2, 256, 6, batch_first=True)] * modes)

        sa = SelfAttention(dim=dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias)
        sa_layers = [sa] * num_layers

        self.cross_attention = CrossAttention(dim=dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias)
        self.self_attention = SelfAttention(dim=dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias)

        self.decoder = nn.Sequential(
                                nn.Linear(256, 2)
        )

        # extract road and vehicle features
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
        # state_components = [batch["past_coordinate"], batch["past_vel"], 
        #                     batch["past_acc"], batch["past_yaw"]]
        
        state_components = [batch["past_coordinate"]]
        
        state = torch.cat(state_components, dim=2)  # (B, 5, 7)

        states = []

        for i in range(self.modes):

            # Pass state data through RNN
            output, h = self.RNNs[i](state) # h stands for hidden state
            output = output[:,-1,:][:,None,:] # (B, 256)
            s = self.decoder(output)
            state = s

            # Cross Attention with road and vehicle features (image features -> hidden state)
            h = self.self_attention(x, h)

            # run RNN for 11 times (6 seconds)
            for _ in range(11):
                output, h = self.RNNs[i](s, h.contiguous())
                s = self.decoder(output)

                state = torch.concat((state, s), dim=1)

            states.append(state)

        states = torch.stack(states, dim=1)
        return states
    
class CrossAttention(nn.Module):
    def __init__(self, dim:int, heads:int, dim_head:int, qkv_bias:bool, norm=nn.LayerNorm):
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

    def forward(self, x, h): # h <- x
        """
        x: (b, d, h, w)
        h: (l, b, d)
        """

        h = rearrange(h, 'l b D -> b l D')          # h: (b, l, d*heads)
        x = rearrange(x, 'b D h w -> b (h w) D')    # x: (b, h*w, d*heads)

        q = self.to_q(h)                            # q: (b, l, d*heads)
        k = self.to_k(x)                            # k: (b, h*w, d*heads)
        v = self.to_v(x)                            # v: (b, h*w, d*heads)

        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # q: (B, l, d)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # k: (B, h*w, d)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # v: (B, h*w, d)
 
        dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)              # dot: (B, l, h*w)
        att = dot.softmax(dim=-1)                                                   # att: (B, l, h*w)

        z = torch.einsum('B Q K, b K d -> B Q d', att, v)                           # a: (B, l, d*heads)
        z = rearrange(z, '(b m) Q d -> b Q (m d)', m=self.heads, d=self.dim_head)   # a: (B, l, d*heads)

        skip = h
        # skip connection (after attention)
        z += skip

        skip = z

        z = self.prenorm(z)
        z = self.mlp(z)

        # skip connection (after mlp) (residual connection)
        z += skip
        z = rearrange(z, 'b l d -> l b d') # (l, b, d)
        return z
    
class SelfAttention(nn.Module):
    def __init__(self, dim:int, heads:int, dim_head:int, qkv_bias:bool, norm=nn.LayerNorm):
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

    def forward(self, x, h): # h <- x
        """
        x: (b, d, h, w)
        h: (l, b, d)
        """
        l, b, D = h.shape
        b, D, _, _ = x.shape
        h = rearrange(h, 'l b D -> b l D')          # h: (b, l, d*heads)
        x = rearrange(x, 'b D h w -> b (h w) D')    # x: (b, h*w, d*heads)
        
        y = torch.concat((h, x), dim=1)             # y: (b, l+h*w, d*heads)

        q = self.to_q(y)                            # q: (b, l, d*heads)
        k = self.to_k(y)                            # k: (b, h*w, d*heads)
        v = self.to_v(y)                            # v: (b, h*w, d*heads)

        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # q: (B, l, d)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # k: (B, h*w, d)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # v: (B, h*w, d)
 
        dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)              # dot: (B, l, h*w)
        att = dot.softmax(dim=-1)                                                   # att: (B, l, h*w)
        z = torch.einsum('B Q K, b K d -> B Q d', att, v)                           # a: (B, l, d*heads)
        z = rearrange(z, '(b m) Q d -> b Q (m d)', m=self.heads, d=self.dim_head)   # a: (B, l, d*heads)
        z = z[:, :l, :]

        skip = h
        # skip connection (after attention)
        z += skip

        skip = z

        z = self.prenorm(z)
        z = self.mlp(z)

        # skip connection (after mlp) (residual connection)
        z += skip
        z = rearrange(z, 'b l d -> l b d') # (l, b, d)
        return z