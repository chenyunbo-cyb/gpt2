from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import time

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


# ------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # q, k, v
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)  # 3*,即生成 Q, K, v 这3个
        # output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.c_proj.NANOGPT_SCALE_INIT = 1
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
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #init params
        self.apply(self._init_weights)
    
    # 初始化神经网络权重
    # 对不同类型的层进行权重和偏置初始化
    def _init_weights(self, module):  
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets = None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T<=self.config.block_size, f"T:{T} is bigger than block_size:{self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  #shape T
        pos_emb = self.transformer.wpe(pos)  # shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # shape (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # shape (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    
    @classmethod  
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # print(sd_keys_hf)
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                # print("-------sd[k].shape-------")
                # print(sd[k].shape)
                # print("------sd_hf[k].shape[::-1]--------")
                # print(sd_hf[k].shape[::-1])
                # print(k)
                # print("             ")
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
# -----------------------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0
        


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T  + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


# -----------------------------------------------------------------------------------------
num_return_sequences = 5
max_length = 30
device = 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
else: device = 'cpu'
# device = 'cpu'
print(f"using device {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=32, T = 128)

# "highest", float32
#  "high", TF32
# "medium", bfloat16
# torch.set_float32_matmul_precision('high') 

model = GPT(GPTConfig())
model.to(device)
# model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
start = time.time()
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(x, y)
        # import code; code.interact(local = locals()) 
    # import code; code.interact(local = locals()) # python调试技巧。停止执行打开交互调试
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T)/(t1 - t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt}ms, tok/sec:{tokens_per_sec:.2f}")
end = time.time()
print("total time：", end-start)

# logits, loss = model(x, y)

# print(logits.shape)
# print(loss)

import sys; sys.exit(0)
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig)
model.eval()
model.to(device)
print(" Yes! First Success! ")


import tiktoken
enc = tiktoken.get_encoding('gpt2')
tockens = enc.encode("Hello, I'm a language model,")
tockens = torch.tensor(tockens, dtype=torch.long) #(8, )
tockens = tockens.unsqueeze(0).repeat(num_return_sequences, 1) #(5, 8)
x = tockens.to(device)


torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:

    with torch.no_grad():
        logits = model(x)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, 1)

        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tockens = x[i, :max_length].tolist()
    decoded = enc.decode(tockens)
    print(">", decoded)