import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class MSA(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_heads = num_heads
        
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.proj = nn.Linear(hidden_dim,hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim,hidden_dim)
        
    def forward(self,x):
        
        bs, all_pixels, c = x.size()
        
        qkv = self.proj(x)
        qkv = qkv.view(bs,all_pixels,3,self.num_heads,self.head_dim)
        qkv = qkv.permute(2,0,3,1,4).contiguous() # (3,bs,num_heads,all_pixels,hidden_dim)
        
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        attn_score = Q @ K.transpose(-1,-2) / self.head_dim ** 0.5
        attn_pobs = F.softmax(attn_score,dim=-1)
        
        out = attn_pobs @ V
        out = out.permute(0,2,1,3).contiguous()
        out = out.view(bs,all_pixels,-1)
        
        out = self.out_proj(out)
        
        return out
    
class MLPBlock(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, attn_type='self', *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.attn_type = attn_type
        
        self.first_norm = nn.LayerNorm(hidden_dim)
        
        self.attention = nn.Identity()
        if attn_type == 'self':
            self.attention = MSA(hidden_dim=hidden_dim, num_heads=num_heads)
        elif attn_type == 'cross':
            self.attention = CrossAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        
        self.second_norm = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4,hidden_dim)
        )
        
    def forward(self,x,tokens=None):
        
        if self.attn_type == 'self':
            x = x + self.attention(self.first_norm(x))
        else:
            x = x + self.attention(self.first_norm(x),tokens)
            
        x = x + self.ffn(self.second_norm(x))
        
        return x

class Stage(nn.Module):
    def __init__(self, *args, hidden_dim=256, num_heads=8, depth=6, attn_type='self', **kwargs):
        super().__init__(*args, **kwargs)
        
        self.attn_type = attn_type
        
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(MLPBlock(hidden_dim=hidden_dim, num_heads=num_heads,attn_type=attn_type))
            
    def forward(self, x, tokens=None):
        
        for layer in self.layers:
            if self.attn_type == 'self':
                x = layer(x)
            else:  # cross
                x = layer(x, tokens)
                
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim,hidden_dim)
        self.k_proj = nn.Linear(hidden_dim,hidden_dim)
        self.v_proj = nn.Linear(hidden_dim,hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self,queries,tokens):
        # queries(Q) - nn.Embedding(num_queries,hidden_dim)
        # tokens(K,V) - output after MSA + pos embed
        
        bs, num_queries, hidden_dim = queries.size()
        _, all_pixels, hidden_dim = tokens.size()
        
        Q = self.q_proj(queries)
        K = self.k_proj(tokens)
        V = self.v_proj(tokens)
        
        Q = Q.view(bs,num_queries,self.num_heads,self.head_dim)
        Q = Q.permute(0,2,1,3).contiguous() # (bs, num_heads, num_queries, head_dim)
        K = K.view(bs,all_pixels,self.num_heads,self.head_dim)
        K = K.permute(0,2,1,3).contiguous() # (bs, num_heads, all_pixels, head_dim)
        V = V.view(bs,all_pixels,self.num_heads,self.head_dim)
        V = V.permute(0,2,1,3).contiguous() # (bs, num_heads, all_pixels, head_dim)
        
        # (bs, num_heads, num_queries, head_dim) @ (bs, num_heads, head_dim, all_pixels) -> (bs, num_heads, num_queries, all_pixels)
        attn_score = Q @ K.transpose(-1,-2) / self.head_dim ** 0.5
        attn_pobs = F.softmax(attn_score,dim=-1)
        
        # (bs, num_heads, num_queries, all_pixels) @ (bs, num_heads, all_pixels, head_dim) -> (bs, num_heads, num_queries, head_dim)
        out = attn_pobs @ V
        out = out.permute(0,2,1,3).contiguous() # (bs, num_queries, num_heads, head_dim)
        out = out.view(bs,num_queries,-1) # (bs, num_queries, hidden_dim)
        
        out = self.out_proj(out)
        
        return out
        

class DETR(nn.Module):
    def __init__(self, img_shape:tuple=(3,224,224), hidden_dim=256, num_heads=8, depth=6, num_queries=200, n_classes=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        self.conv = nn.Conv2d(512,hidden_dim,1)
        
        self.attention = Stage(hidden_dim=hidden_dim,num_heads=num_heads,depth=depth,attn_type='self')
        
        all_pixels = (img_shape[1] // 32) * (img_shape[2] // 32)
        self.positions = nn.Parameter(torch.randn(1,all_pixels,hidden_dim))
        
        self.query_embed = nn.Embedding(num_queries,hidden_dim)
        self.cross_attention = Stage(hidden_dim=hidden_dim,num_heads=num_heads,depth=depth,attn_type='cross')
        
        self.class_head = nn.Linear(hidden_dim,n_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4,hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim,4),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        
        x = self.backbone(x)
        x = self.conv(x)
        
        bs, c, h, w = x.size()
        
        x = x.view(bs,c,-1)
        x = x.permute(0,2,1).contiguous() # (bs,all_pixels,hidde_dim)
        
        x = self.attention(x) # (bs,all_pixels,hidde_dim)
        x = x + self.positions
        
        queries = self.query_embed.weight.unsqueeze(0).repeat(bs,1,1)
        queries  = self.cross_attention(queries,x)
        
        classes = self.class_head(queries)
        bboxes = self.bbox_head(queries)
        
        return classes, bboxes