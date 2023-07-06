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
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert d_model == self.head_dim * n_head, "d_model must be divisible by n_head"

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.out_fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None):
        # batch_size, seq_len, d_model
        b1, l1, d1 = q.shape
        b2, l2, d2 = k.shape
        b3, l3, d3 = v.shape

        assert b1 == b2 == b3, "Must be same batchsize btw q,k,v"
        assert l2 == l3, "Must be same sequence length btw k,v"
        assert d1 == d2 == d3, "Must be same embedding size btw q,k,v"

        b = b1

        q = self.wq(q)
        k = self.wq(k)
        v = self.wq(v)

        # (batch_size * n_head), seq_len, head_dim
        q = q.view(b * self.n_head, l1, self.head_dim)
        k = k.view(b * self.n_head, l2, self.head_dim)
        v = v.view(b * self.n_head, l2, self.head_dim)

        # (batch_size * n_head), seq_len, seq_len
        attn_score = torch.bmm(q, k.transpose(1, 2))
        scaled_attn_score = attn_score / math.sqrt(self.head_dim)
        if attn_mask is not None:
            mask = (attn_mask==float('-inf')).unsqueeze(0).repeat(scaled_attn_score.size(0), 1, 1)
            scaled_attn_score[mask] = 0
        soft_max_attn = F.softmax(scaled_attn_score, dim=-1)
        attn_value = torch.bmm(soft_max_attn, v).view(b, l1, self.d_model)
        out = self.out_fc(attn_value)

        return out
    
class ASLTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.self_attn(x, x, x))
        x = self.norm2(x + self.ff2(self.act(self.ff1(x))))
        return x
    
class ASLTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, n_layers, d_ff):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(ASLTransformerEncoderLayer(d_model, n_head, d_ff))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ASLTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.en_dec_attn = MultiheadAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, k, v, attn_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, attn_mask))
        x = self.norm2(x + self.en_dec_attn(x, k, v))
        x = self.norm3(x + self.ff2(self.act(self.ff1(x))))
        return x

class ConstantInput(nn.Module):
    def __init__(self, y_dim, d_model):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, y_dim, d_model))

    def forward(self, x):
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1)

        return out
    
class ASLTransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, n_layers, d_ff):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(ASLTransformerDecoderLayer(d_model, n_head, d_ff))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, r, attn_mask=None):
        for layer in self.layers:
            x = layer(x, r, r, attn_mask)
        return x

class ASLTransformer(nn.Module):
    def __init__(self, y_dim, y_len, d_model, n_head, n_enc_layers, n_dec_layers, d_ff):
        super().__init__()
        self.constant = ConstantInput(y_len, d_model)
        self.encoder = ASLTransformerEncoder(d_model, n_head, n_enc_layers, d_ff)
        self.decoder = ASLTransformerDecoder(d_model, n_head, n_dec_layers, d_ff)
        self.fc_out = nn.Linear(d_model, y_dim)

    def forward(self, x, attn_mask=None):
        out_lst = self.encoder(x)
        c = self.constant(x)
        out = self.decoder(c, out_lst, attn_mask)
        out = self.fc_out(out)
        out = F.softmax(out, dim=-1)
        return out

    @staticmethod
    def generate_mask(size):
        mask = torch.triu(torch.full((size, size), float('-inf')), diagonal=1)
        mask.requires_grad = False
        return mask

if __name__ == "__main__":
    model = ASLTransformer(
        y_dim=62, y_len=64, d_model=152, n_head=8, n_enc_layers=6, n_dec_layers=1, d_ff=512
    ).to('cuda')

    # batch_size, seq_len, d_model
    x = torch.randn(16, 512, 152, device='cuda')
    y = torch.randint(low=0, high=62, size=(16, 64), device='cuda')
    
    print_tensor(x, desc='x')
    print_tensor(y, desc='y')

    out = model(x, y)

    print_tensor(out, desc='out')