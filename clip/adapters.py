import torch
from torch import nn
import torch.nn.functional as F



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class PFCrossAttention(nn.Module):
    """
        Parameter-free Cross-Attention
        https://arxiv.org/abs/2209.14169
    """
    def __init__(self, dim, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.scale = qk_scale or dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, kv):
        k, v = kv, kv

        attn = (query @ k.transpose(1,2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query = attn @ v
        query = self.proj_drop(query)

        return query
    


class LocalAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, kernel_size=3, text_dim=512):
        super().__init__()
        hidden_dim = text_dim
        layers1 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=stride, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=3//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        ]
        self.conv_adapter_layers2 = nn.Sequential(*layers2)

        self.conv_adapter_layers = nn.Conv2d(2, 2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.conv_adapter_final = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=stride, padding=1//2, bias=False)

        self.adapter_norm_q1 = nn.LayerNorm(text_dim)
        self.adapter_norm_q2 = nn.LayerNorm(text_dim)
        self.adapter_norm_kv = nn.LayerNorm(text_dim)
        self.scale = text_dim ** -0.5
        self.adapter_cattn = PFCrossAttention(text_dim)

    def forward(self, x, text_fea=None):
        # x: channel first
        x_cls, x = x[:1], x[1:]     
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

        # Dual convolutional streams
        x_loc1 = self.conv_adapter_layers1(x_loc)
        x_loc2 = self.conv_adapter_layers2(x_loc)
        x_loc1 = x_loc1.reshape(B, -1, tok)
        x_loc2 = x_loc2.reshape(B, -1, tok)

        # Cross attention with text features
        x_loc1 = x_loc1 + self.adapter_cattn(
            self.adapter_norm_q1(x_loc1.permute(0, 2, 1)), 
            self.adapter_norm_kv(text_fea.permute(1, 0, 2))
        ).permute(0, 2, 1)
        x_loc2 = x_loc2 + self.adapter_cattn(
            self.adapter_norm_q2(x_loc2.permute(0, 2, 1)), 
            self.adapter_norm_kv(text_fea.permute(1, 0, 2))
        ).permute(0, 2, 1)

        # Reshape and concat
        x_loc1 = x_loc1.reshape(B, -1, H, H)
        x_loc2 = x_loc2.reshape(B, -1, H, H)
        x_loc = torch.cat([x_loc1, x_loc2], dim=1)

        # Max and Average pooling
        avg_x = torch.mean(x_loc, dim=1, keepdim=True)  
        max_x, _ = torch.max(x_loc, dim=1, keepdim=True)

        # Aggregated convolution
        agg = torch.cat([avg_x, max_x], dim=1)
        y = self.conv_adapter_layers(agg)
        y = F.sigmoid(y)

        # Final multiplication and convolution
        x = x_loc1 * y[:, 0].unsqueeze(1) + x_loc2 * y[:, 1].unsqueeze(1)
        x = self.conv_adapter_final(x)
        x = x.reshape(B, -1, tok).permute(2, 0, 1)  
        x = torch.cat([x_cls, x], dim=0)

        return x

