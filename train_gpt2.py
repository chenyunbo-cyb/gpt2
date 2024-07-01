import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # q, k, v
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)  # 3*,即生成 Q, K, v 这3个
        # output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

         # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k , v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        #output projection
        y = self.c_proj(y)
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        #将其输入的每一个嵌入向量（维度为config.n_embd），映射为一个更大的中间 Vector其维度为4*config.n_embd。
        # 这相当于扩展了模型可以表示和学习的特征空间。 计算更高维度的特征表示，以便在此基础上进一步计算或学习。
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) 
        # self.gelu = nn.GELU(approximate='tanh')
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        # x = self.dropout(x)
        return x



class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) #aggregation/pooling/weighted-sum/reduce-operation fun, communicate, exchange information
        x = x + self.mlp(self.ln_2(x)) # MLP is the map 映射
        return x
    


@dataclass
class GPTConfig:
    # gpt2-124M
    block_size: int = 1024    # 256  # max sequence length
    vocab_size: int = 50257   # 65   # number of tockens # 50,000 BPE merges + 256 bytes tockens + 1 <|endoftext|>tocken
    n_layer: int = 12         # 6    # number of layers
    n_head: int = 12          # 6    # number of heads
    n_embd: int = 768         # 384  # embedding dimension

    # dropout: float = 0.0
    # bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self ,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入层，tocken embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),  # 位置编码，输出将进入transformer
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  ## n = n_layer层
            ln_f = nn.LayerNorm(config.n_embd), # gpt2新增的层
        ))
        # 线性层，最终的分类器
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias= False)