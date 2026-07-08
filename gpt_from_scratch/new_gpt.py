# %%
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.token_embed = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embed = nn.Embedding(config["context_length"] , config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.trf_block = nn.Sequential(
            *[DummyTransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"] , config["vocab_size"],bias=False)
        
    def forward(self,in_idx):
        batch_size, seq_len = in_idx.shape
        token_embed =self.token_embed(in_idx)
        pos_embed =self.pos_embed(torch.arange(seq_len))
        
        x= token_embed+ pos_embed
        # print(f" shape of embedding : {x.shape}")
        
        x= self.drop_emb(x)
        x= self.trf_block(x)
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        
        
        


class DummyTransformerBlock(nn.Module):
    def __init__(self ,cfg ):
        super().__init__()
        self.attention = MultiheadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = DummyLayerNorm(cfg["emb_dim"])
        self.norm2 = DummyLayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x):
        shortcut =x
        x= self.norm1(x)
        x= self.attention(x)
        x= self.drop_shortcut(x)
        x =x+shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x
    
        
        
        
        
        
    

class DummyLayerNorm(nn.Module):
    def __init__(self , emb_dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
        
    def forward(self , x):
        mean = x.mean(dim =-1 , keepdim = True) 
        var = x.var(dim =-1 , keepdim = True ,unbiased= False) 
        norm_x =  (x-mean)/torch.sqrt(var + self.eps)
        
        
        return self.scale * norm_x + self.shift
    

class GELU(nn.Module) :
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
            ))
        
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"] , 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear( 4 * cfg["emb_dim"] , cfg["emb_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)


class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length, dropout, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out,
        self.head_dim = d_out//num_heads
        
        self.d_in = d_in
        self.k_weights = nn.Linear(d_in,d_out , bias )
        self.q_weights = nn.Linear(d_in,d_out , bias )
        self.v_weights = nn.Linear(d_in,d_out , bias )
        self.proj = nn.Linear(d_out,d_out)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('multi_mask' , torch.triu(torch.ones(context_length,context_length), diagonal=1))
        
    def forward(self,x):
        b, num_tokens, d_in = x.shape
        keys= self.k_weights(x)
        queries= self.q_weights(x)
        values= self.v_weights(x)
        
        
        
        keys = keys.view(b,num_tokens , self.num_heads, self.head_dim)
        queries = queries.view(b,num_tokens , self.num_heads, self.head_dim)
        values = values.view(b,num_tokens , self.num_heads, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        
        
        attention = queries @ keys.transpose(-1,-2)
        # return attention
        
        mask_bool = self.multi_mask.bool()[:num_tokens, :num_tokens]
        masked_attention = attention.masked_fill( mask_bool ,-torch.inf)
        atten_soft = torch.softmax(masked_attention/keys.shape[-1]**0.5 , dim=-1)
        # print(f"attention soft shape {atten_soft.shape , values.shape}")
        
        
        context_vec = (atten_soft @ values).transpose(1,2)
        
        context_vec = context_vec.contiguous().view(b,num_tokens,-1)
        context_vec = self.proj(context_vec)
        
        return context_vec
        
     

# %%
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
        "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

# %%
import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_embed.weight = assign(gpt.pos_embed.weight, params['wpe'])
    gpt.token_embed.weight = assign(gpt.token_embed.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_block[b].attention.q_weights.weight = assign(
        gpt.trf_block[b].attention.q_weights.weight, q_w.T)
        gpt.trf_block[b].attention.k_weights.weight = assign(
        gpt.trf_block[b].attention.k_weights.weight, k_w.T)
        gpt.trf_block[b].attention.v_weights.weight = assign(
        gpt.trf_block[b].attention.v_weights.weight, v_w.T)
        q_b, k_b, v_b = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_block[b].attention.q_weights.bias = assign(
        gpt.trf_block[b].attention.q_weights.bias, q_b)
        gpt.trf_block[b].attention.k_weights.bias = assign(
        gpt.trf_block[b].attention.k_weights.bias, k_b)
        gpt.trf_block[b].attention.v_weights.bias = assign(
        gpt.trf_block[b].attention.v_weights.bias, v_b)
        gpt.trf_block[b].attention.proj.weight = assign(
        gpt.trf_block[b].attention.proj.weight,
        params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_block[b].attention.proj.bias = assign(
        gpt.trf_block[b].attention.proj.bias,
        params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_block[b].ff.layers[0].weight = assign(
        gpt.trf_block[b].ff.layers[0].weight,
        params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_block[b].ff.layers[0].bias = assign(
        gpt.trf_block[b].ff.layers[0].bias,
        params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_block[b].ff.layers[2].weight = assign(
        gpt.trf_block[b].ff.layers[2].weight,
        params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_block[b].ff.layers[2].bias = assign(
        gpt.trf_block[b].ff.layers[2].bias,
        params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_block[b].norm1.scale = assign(
        gpt.trf_block[b].norm1.scale,
        params["blocks"][b]["ln_1"]["g"])
        gpt.trf_block[b].norm1.shift = assign(
        gpt.trf_block[b].norm1.shift,
        params["blocks"][b]["ln_1"]["b"])
        gpt.trf_block[b].norm2.scale = assign(
        gpt.trf_block[b].norm2.scale,
        params["blocks"][b]["ln_2"]["g"])
        gpt.trf_block[b].norm2.shift = assign(
        gpt.trf_block[b].norm2.shift,
        params["blocks"][b]["ln_2"]["b"])
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# %%



