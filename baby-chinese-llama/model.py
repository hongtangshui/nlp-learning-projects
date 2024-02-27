import math
from dataclasses import dataclass
import struct
import inspect
from typing import Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.1

class RMSNorm(torch.nn.Module):
    '''
    大约等于layernorm没有偏移
    x / sqrt(x^2)
    '''
    def __init__(self, dim:int, eps: float):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        '''
        x: (bs, sl, dim)
        '''
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # TODO: 这一部分GPU算?
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def apply_rotary_emb( xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1])    

def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0 ,dim, 2)[: dim//2].float() / dim))
    t = torch.arange(end, device=freqs.device)  # 时间维度
    freqs = torch.outer(t, freqs).float()   # 外积 第i,j 元素为 t[i] * freqs[j]
    freqs_cos = torch.cos(freqs)    # real part
    freqs_sin = torch.sin(freqs)    # img part
    return freqs_sin, freqs_cos

def reshape_for_boardcaset(freqs_cis: torch.Tensor, x:torch.Tensor):
    '''
    freqs_sin: torch.Tensor (sl, dim/2)
    '''
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim-1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    '''
    xq: (bs, sl, h)
    '''
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_boardcaset(freqs_cos, xq_r)
    freqs_sin = reshape_for_boardcaset(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimenstions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xq_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    '''
    x: torch.Tensor (bs, sl, n_kv_heads, head_dim)
    torch.repeat_interleave(x, dim=2, repeats=n_rep)
    '''
    bs, sl, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, sl, n_kv_heads, n_rep, head_dim).reshape(bs, sl, n_kv_heads*n_rep, head_dim)


class Attention(nn.Module):
    def __init__(self, args:ModelArgs):
        # TODO: 为什么Attention这部分需要分布式??
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size=1

        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = args.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
    
        self.dim=args.dim
        self.n_heads=args.n_heads
        self.head_dim=self.dim//self.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
    
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        bsz, sl, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, sl, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, sl, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, sl, self.n_local_kv_heads, self.head_dim)
        # RoPE relative posotional embedding
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expaned out keys and values
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        
        xq = xq.transpose(1, 2) # (bs, n_heads, sl, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)    # [bs, n_heads, sl, sl]
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, :sl, :sl]
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)
        
        output = output.transpose(1, 2).contiguous().view(bsz, sl, -1)

        output = self.wo(output)
        output = self.resid_dropout(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # 11008
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id=layer_id
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)
    
    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # embedding
        self.tok_embedding = nn.Embedding(self.vocab_size, self.params.dim)

        # pre compute for RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # transformer
        self.layers = nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, self.params))
        
        # dropout
        self.dropout = nn.Dropout(params.dropout)
        # output
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding paramters with the embedding paramters
        self.tok_embedding.weight = self.output.weight # # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self.__init__weights)

    def __init__weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.weight)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen = tokens.shape  # (b, l) 每个元素是对应token的index
        h = self.tok_embedding(tokens)
        h = self.dropout(h)

        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if target is not None:
            # teacher forcing只需要一次前向计算和反向传播就可以
            logits = self.output(h) # (bsz, sl, vsz)
            # target (bsz, sl)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        else:
            # inference time
            # h (bsz, sl, dim)
            logits = self.output(h[:, [-1],:])
            self.last_loss = None
        return logits
    
    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        '''
        这个函数的作用就是创建一个optimizer
        '''
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # TODO param_dict.items()??
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        # num 
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        

        fused_available='fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type=='cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        # TODO whatis fused adam
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    
    def estimate_mfu(self, fwdbwd_iter, dt):
        """estimated model floats utilization in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        return 0
    
    @torch.no_grad()
    def generate(self, idx, eos, max_new_tokens, temperature=1.0, top_k=None):
        '''
        take a conditioning sequence of indices idx (LongTensor of shape(b, t)) and complete the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Input: 
            idx @torch.tensor (b, t)
        '''
        for _ in range(max_new_tokens):
            idx_cond = idx if dix.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            logits = self(idx_cond) # (b, t, h)
            logits = logits[:, -1, :]   # 自回归中最后一个logits分布，softmax之后就是对应到此表的下个token分布
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to normalized probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(prods, num_samples=1)
            idx=torch.cat((idx, idx_next), dim=1)
            if idx_net==eos:
                break
        return idx
    
    def export(self, filepath='model.bin'):
        f=open(filepath, 'wb')

        def serialize(t):
            d = t.detach().cpu().view(-1).numpy().astype(np.float32)
            b = struct.pack(f'{len(d)}f', *d)
            f.write(b)
        
        # first write out the header
        hidden_dim = self.layers[0].feed_forward.w1.weight.shape[0]
        p = self.params
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack('iiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads, n_kv_heads, p.vocab_size, p.max_seq_len)
        f.write(header)

        # next write out the embedding weights
        serialize(self.tok_embeddings.weight)

        # now all the layers
        # attention weights
        for layer in self.layers:
            serialize(layer.attention_norm.weight)
        for layer in self.layers:
            serialize(layer.attention.wq.weight)
        for layer in self.layers:
            serialize(layer.attention.wk.weight)
        for layer in self.layers:
            serialize(layer.attention.wv.weight)
        for layer in self.layers:
            serialize(layer.attention.wo.weight)
        for layer in self.layers:
            serialize(layer.fnn_norm.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w1.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w1.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w2.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w3.weight)
        # final norm
        serialize(self.norm.weight)
        # feqs_cis
        serialize(self.freqs_cos[:p.max_seq_len])
        serialize(self.freqs_sin[:p.max_seq_len])

        f.close()
        print(f"wrote {filepath}")

if __name__ == "__main__":
    precompute_freqs_cis(256, 1024)