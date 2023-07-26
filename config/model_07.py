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

class EncoderConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout):
        super().__init__()
        # conv
        self.conv1d_1 = nn.Conv1d(dim_in, dim_out, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(dim_out, dim_out, kernel_size=3, padding=1, stride=2)
        # norm
        self.norm1 = nn.BatchNorm1d(dim_in)
        self.norm2 = nn.BatchNorm1d(dim_out)
        # act
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(dim_in, dim_out, kernel_size=1, padding=0, stride=2)
    
    def forward(self, x):
        out1 = self.conv1d_1(self.act1(self.norm1(x)))
        out2 = self.conv1d_2(self.act2(self.norm2(out1)))
        out = self.downsample(x) + out2
        return out, out1

class DecoderConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout):
        super().__init__()
        # conv
        self.conv1d_1 = nn.ConvTranspose1d(dim_in, dim_out, kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(dim_out*2, dim_out, kernel_size=3, padding=1)
        # norm
        self.norm1 = nn.BatchNorm1d(dim_in)
        self.norm2 = nn.BatchNorm1d(dim_out * 2)
        # act
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.upsample = nn.ConvTranspose1d(dim_in, dim_out, kernel_size=2, stride=2)
    
    def forward(self, x, r):
        out = self.dropout1(self.conv1d_1(self.act1(self.norm1(x))))
        out = torch.cat([out, r], dim=1)
        out = self.dropout2(self.conv1d_2(self.act2(self.norm2(out))))
        out = self.upsample(x) + out
        return out
    

class ASLModel_(nn.Module):
    def __init__(self, x_dim, x_len, y_dim, y_len, d_model, nlayers, dropout):
        super().__init__()
        self.fc_in = nn.Linear(x_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True, norm_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        enc_layer = []
        for i in range(nlayers):
            enc_layer.append(EncoderConvLayer(x_len, x_len, dropout))
        self.enc_layer = nn.Sequential(*enc_layer)

        dec_layer = []
        for i in range(nlayers):
            dec_layer.append(DecoderConvLayer(x_len, x_len, dropout))
        self.dec_layer = nn.Sequential(*dec_layer)

        self.conv_out = nn.Conv1d(x_len, y_len, kernel_size=1)
        self.fc_out = nn.Linear(d_model, y_dim)

    def forward(self, src):
        # b, L, x_D
        out = self.fc_in(src) # b, L, D
        residuals = []
        for layer in self.enc_layer:
            out, r = layer(out)
            residuals.insert(0, r)

        for layer, r in zip(self.dec_layer, residuals):
            out = layer(out, r)

        out = self.conv_out(out)
        out = self.fc_out(out)
        return out
    
    def inference(self, src):
        out = self.forward(src)
        out = torch.argmax(out, dim=-1)
        return out
    
ASLModel = ASLModel_(
    x_dim=84, x_len=512, y_dim=62, y_len=64, d_model=256, nlayers=5, dropout=0.3
)

if __name__ == "__main__":
    model = ASLModel.to('cuda')
    bs = 16
    x_len = 512
    x_dim = 84
    y_len = 64
    y_dim = 62
    x = torch.randn(bs, x_len, x_dim, device='cuda')
    y = torch.randint(low=0, high=y_dim, size=(bs, y_len), device='cuda')
    
    print_tensor(x, desc='x')
    print_tensor(y, desc='y')

    out = model(x)
    infer = model.inference(x)

    print_tensor(out, desc='out')
    print_tensor(infer, desc='infer')