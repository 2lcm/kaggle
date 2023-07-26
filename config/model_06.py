import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def print_tensor(x, desc=""):
    if desc:
        print(f"{desc}: {x.shape} {x.dtype} {x.min()} {x.max()}")
    else:
        print(f"{x.shape} {x.dtype} {x.min()} {x.max()}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
        
class ASLModel_(nn.Module):
    def __init__(self, x_dim, x_len, y_dim, y_len, d_model, nhead, d_hid, nlayers, dropout):
        super().__init__()
        self.fc_in = nn.Linear(x_dim, d_model)
        self.embedding = nn.Embedding(y_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc_out_1 = nn.Linear(x_len, y_len)
        self.fc_out_2 = nn.Linear(d_model, y_dim)
        self.y_len = y_len


    def forward(self, src):
        src = self.pos_encoder(self.fc_in(src))
        out = self.transformer_encoder(src)
        out = out.permute(0, 2, 1)
        out = self.fc_out_1(out)
        out = out.permute(0, 2, 1)
        out = self.fc_out_2(out)
        return out
    
    def inference(self, src):
        out = self.forward(src)
        out = torch.argmax(out, dim=-1)
        return out
    
ASLModel = ASLModel_(
    x_dim=84, x_len=512, y_dim=62, y_len=64, d_model=128, nhead=1, d_hid=512, nlayers=12, dropout=0.1
)

if __name__ == "__main__":
    model = ASLModel.to('cuda')
    bs = 16
    y_len = 64
    x_len = 512
    x_dim = 84
    y_dim = 62
    x = torch.randn(bs, x_len, x_dim, device='cuda')
    y = torch.randint(low=0, high=y_dim, size=(bs, y_len), device='cuda')
    
    print_tensor(x, desc='x')
    print_tensor(y, desc='y')

    out = model(x)