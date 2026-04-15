import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block as TimmBlock
import numbers

DTCWT_COEFFS = {
    'level1': {
        'h0a': [-0.0883883476, 0.0883883476, 0.695879989, 0.695879989, 0.0883883476, -0.0883883476],
        'h1a': [-0.011226792, 0.011226792, 0.0883883476, 0.0883883476, -0.695879989, 0.695879989, -0.0883883476, -0.0883883476, 0.011226792, -0.011226792],
        'h0b': [0.011226792, 0.011226792, -0.0883883476, 0.0883883476, 0.695879989, 0.695879989, 0.0883883476, -0.0883883476, 0.011226792, 0.011226792],
        'h1b': [0.0883883476, 0.0883883476, -0.695879989, 0.695879989, -0.0883883476, -0.0883883476],
    },
    
    'qshift': {
        'h0a': [0.00034062, -0.00023787, -0.00349111, 0.00263218, 0.02178652, -0.00277971, 
                -0.09840428, 0.0582554, 0.4540297, 0.75798424, 0.40940683, -0.04303767, -0.02825287, 0.00726402],
        'h1a': [-0.00726402, -0.02825287, 0.04303767, 0.40940683, -0.75798424, 0.4540297, 
                -0.0582554, -0.09840428, -0.00277971, -0.02178652, 0.00263218, 0.00349111, -0.00023787, -0.00034062],
    }
}

def get_dtcwt_filters(level_idx, device='cpu', dtype=torch.float32):
    coeffs = DTCWT_COEFFS
    
    if level_idx == 0:
        c = coeffs['level1']
        h0a, h1a = c['h0a'], c['h1a']
        h0b, h1b = c['h0b'], c['h1b']
    else:
        c = coeffs['qshift']
        h0a_list = c['h0a']
        h1a_list = c['h1a']
        h0b_list = h0a_list[::-1]
        h1b_list = h1a_list[::-1]
        
        h0a, h1a = h0a_list, h1a_list
        h0b, h1b = h0b_list, h1b_list

    return (
        torch.tensor(h0a, device=device, dtype=dtype).view(1, 1, -1),
        torch.tensor(h1a, device=device, dtype=dtype).view(1, 1, -1),
        torch.tensor(h0b, device=device, dtype=dtype).view(1, 1, -1),
        torch.tensor(h1b, device=device, dtype=dtype).view(1, 1, -1)
    )

class DTCWTLayer(nn.Module):
    def __init__(self, level=1):
        super().__init__()
        self.level = level
        self.dummy_param = nn.Parameter(torch.empty(0)) 

    def _apply_conv_1d_dim(self, x, filt, stride, dim='h'):
        k = filt.shape[2] if dim == 'h' else filt.shape[3]
        
        pad_total = k - 1
        pad_start = pad_total // 2
        pad_end = pad_total - pad_start
        
        if dim == 'h':
            x_pad = F.pad(x, (0, 0, pad_start, pad_end), mode='reflect')
            return F.conv2d(x_pad, filt, stride=stride, groups=x.shape[1])
        else:
            x_pad = F.pad(x, (pad_start, pad_end, 0, 0), mode='reflect')
            return F.conv2d(x_pad, filt, stride=stride, groups=x.shape[1])

    def _convolve_2d_tree(self, x, lo, hi):
        C = x.shape[1]
        
        filt_lo_r = lo.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        filt_hi_r = hi.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        
        l_rows = self._apply_conv_1d_dim(x, filt_lo_r, stride=(2, 1), dim='h')
        h_rows = self._apply_conv_1d_dim(x, filt_hi_r, stride=(2, 1), dim='h')
        
        filt_lo_c = lo.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        filt_hi_c = hi.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        
        ll = self._apply_conv_1d_dim(l_rows, filt_lo_c, stride=(1, 2), dim='w')
        lh = self._apply_conv_1d_dim(l_rows, filt_hi_c, stride=(1, 2), dim='w')
        hl = self._apply_conv_1d_dim(h_rows, filt_lo_c, stride=(1, 2), dim='w')
        hh = self._apply_conv_1d_dim(h_rows, filt_hi_c, stride=(1, 2), dim='w')
        
        if ll.shape != lh.shape or ll.shape != hl.shape:
             min_h = min(ll.shape[2], hl.shape[2])
             min_w = min(ll.shape[3], lh.shape[3])
             ll = ll[:, :, :min_h, :min_w]
             lh = lh[:, :, :min_h, :min_w]
             hl = hl[:, :, :min_h, :min_w]
             hh = hh[:, :, :min_h, :min_w]

        return torch.cat([ll, lh, hl, hh], dim=1)

    def forward(self, x):
        curr_x = x
        device = x.device
        dtype = x.dtype
        
        for i in range(self.level):
            h0a, h1a, h0b, h1b = get_dtcwt_filters(i, device, dtype)
            out_a = self._convolve_2d_tree(curr_x, h0a, h1a)
            out_b = self._convolve_2d_tree(curr_x, h0b, h1b)
            curr_x = torch.cat([out_a, out_b], dim=1)
            
        return curr_x

class IDTCWTLayer(nn.Module):
    def __init__(self, level=1):
        super().__init__()
        self.level = level
        self.dummy_param = nn.Parameter(torch.empty(0))

    def _apply_conv_trans_1d_dim(self, x, filt, stride, dim='h', target_size=None):
        k = filt.shape[2] if dim == 'h' else filt.shape[3]
        pad_total = k - 1
        pad_start = pad_total // 2
        pad_end = pad_total - pad_start
        
        if dim == 'h':
            out = F.conv_transpose2d(x, filt, stride=stride, groups=x.shape[1], padding=0)
            
            if pad_start > 0 or pad_end > 0:
                h_raw = out.shape[2]
                out = out[:, :, pad_start : h_raw - pad_end, :]
        else:
            out = F.conv_transpose2d(x, filt, stride=stride, groups=x.shape[1], padding=0)
            if pad_start > 0 or pad_end > 0:
                w_raw = out.shape[3]
                out = out[:, :, :, pad_start : w_raw - pad_end]
                
        return out

    def _inv_convolve_2d_tree(self, x_stacked, lo, hi):
        B, C4, H, W = x_stacked.shape
        C = C4 // 4
        ll, lh, hl, hh = torch.split(x_stacked, C, dim=1)
        
        target_h = H * 2
        target_w = W * 2

        f_lo_c = lo.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        f_hi_c = hi.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        f_lo_r = lo.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        f_hi_r = hi.view(1, 1, -1, 1).repeat(C, 1, 1, 1)

        l_rows = self._apply_conv_trans_1d_dim(ll, f_lo_c, (1, 2), 'w') + \
                 self._apply_conv_trans_1d_dim(lh, f_hi_c, (1, 2), 'w')
                 
        h_rows = self._apply_conv_trans_1d_dim(hl, f_lo_c, (1, 2), 'w') + \
                 self._apply_conv_trans_1d_dim(hh, f_hi_c, (1, 2), 'w')
        
        if l_rows.shape[3] != target_w:
            l_rows = F.interpolate(l_rows, size=(H, target_w), mode='nearest')
            h_rows = F.interpolate(h_rows, size=(H, target_w), mode='nearest')

        out = self._apply_conv_trans_1d_dim(l_rows, f_lo_r, (2, 1), 'h') + \
              self._apply_conv_trans_1d_dim(h_rows, f_hi_r, (2, 1), 'h')
              
        if out.shape[2] != target_h:
            out = F.interpolate(out, size=(target_h, target_w), mode='nearest')

        return out

    def forward(self, x):
        curr_x = x
        device = x.device
        dtype = x.dtype
        
        for i in reversed(range(self.level)):
            h0a, h1a, h0b, h1b = get_dtcwt_filters(i, device, dtype)
            
            channels = curr_x.shape[1]
            half_c = channels // 2
            
            tree_a_in = curr_x[:, :half_c, :, :]
            tree_b_in = curr_x[:, half_c:, :, :]
            
            out_a = self._inv_convolve_2d_tree(tree_a_in, h0a, h1a)
            out_b = self._inv_convolve_2d_tree(tree_b_in, h0b, h1b)
            
            curr_x = (out_a + out_b) / 2
            
        return curr_x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0., drop_path=0.):
        super().__init__()
        
        if dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({dim}) must be divisible by number of heads ({num_heads}).")

        self.block = TimmBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True, 
            proj_drop=dropout,
            attn_drop=dropout,
            drop_path=drop_path
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = x.flatten(2).transpose(1, 2)

        x = self.block(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x

class DWTLayer(nn.Module):
    def __init__(self, wave='haar', level=1):
        super().__init__()
        self.level = level
        
        try:
            w = pywt.Wavelet(wave)
        except ValueError:
            w = pywt.Wavelet('haar')

        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)

        filt_ll = torch.outer(dec_lo, dec_lo)
        filt_lh = torch.outer(dec_lo, dec_hi)
        filt_hl = torch.outer(dec_hi, dec_lo)
        filt_hh = torch.outer(dec_hi, dec_hi)
        
        filters = torch.stack([filt_ll, filt_lh, filt_hl, filt_hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)

        self.filt_len = filters.shape[-1]
        self.pad_size = (self.filt_len - 2) // 2

    def forward(self, x):
        current_x = x
        for _ in range(self.level):
            b, c, h, w = current_x.shape
            
            filters = self.filters.repeat(c, 1, 1, 1)

            if self.pad_size > 0:
                current_x = F.pad(current_x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='reflect')
            
            current_x = F.conv2d(current_x, filters, stride=2, groups=c)
            
        return current_x

class IWTLayer(nn.Module):
    def __init__(self, wave='haar', level=1):
        super().__init__()
        self.level = level
        
        try:
            w = pywt.Wavelet(wave)
        except ValueError:
            w = pywt.Wavelet('haar')

        rec_hi = torch.tensor(w.rec_hi[::-1], dtype=torch.float32)
        rec_lo = torch.tensor(w.rec_lo[::-1], dtype=torch.float32)

        filt_ll = torch.outer(rec_lo, rec_lo)
        filt_lh = torch.outer(rec_lo, rec_hi)
        filt_hl = torch.outer(rec_hi, rec_lo)
        filt_hh = torch.outer(rec_hi, rec_hi)

        filters = torch.stack([filt_ll, filt_lh, filt_hl, filt_hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)

        self.filt_len = filters.shape[-1]
        self.pad = (self.filt_len - 2) // 2

    def forward(self, x):
        current_x = x
        
        for _ in range(self.level):
            b, c_in, h, w = current_x.shape
            c_out = c_in // 4
            
            filters = self.filters.repeat(c_out, 1, 1, 1)

            current_x = F.conv_transpose2d(
                current_x, 
                filters, 
                stride=2, 
                groups=c_out, 
                padding=self.pad
            )
            
        return current_x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size 
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1)) 
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
                
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size) 
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask) 

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class SwinTransformerPairBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.original_resolution = input_resolution
        
        H, W = input_resolution
        
        self.pad_r = (window_size - W % window_size) % window_size
        self.pad_b = (window_size - H % window_size) % window_size
        
        H_padded = H + self.pad_b
        W_padded = W + self.pad_r
        
        self.padded_resolution = (H_padded, W_padded)

        self.block1 = SwinTransformerBlock(
            dim=dim, 
            input_resolution=self.padded_resolution,
            num_heads=num_heads, 
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path
        )
        
        self.block2 = SwinTransformerBlock(
            dim=dim, 
            input_resolution=self.padded_resolution,
            num_heads=num_heads, 
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, self.pad_r, 0, self.pad_b))
        
        x = x.flatten(2).transpose(1, 2)
        
        x = self.block1(x)
        x = self.block2(x)
        
        x = x.transpose(1, 2).view(B, C, self.padded_resolution[0], self.padded_resolution[1])
        
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :, :H, :W]
            
        return x

def _restormer_to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def _restormer_to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class RestormerBiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class RestormerWithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class RestormerLayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = RestormerBiasFreeLayerNorm(dim)
        else:
            self.body = RestormerWithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return _restormer_to_4d(self.body(_restormer_to_3d(x)), h, w)

class RestormerFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class RestormerAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = RestormerLayerNorm(dim, LayerNorm_type)
        self.attn = RestormerAttention(dim, num_heads, bias)
        self.norm2 = RestormerLayerNorm(dim, LayerNorm_type)
        self.ffn = RestormerFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class EfficientSHMA(nn.Module):
    def __init__(self, dim, qk_dim=None, p_dim=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim or dim // 2
        self.scale = self.qk_dim ** -0.5

        self.to_q = nn.Conv2d(dim, self.qk_dim, 1, bias=True)
        self.to_k = nn.Conv2d(dim, self.qk_dim, 1, bias=True)

        mid_dim = p_dim or dim
        self.to_v_gate = nn.Conv2d(dim, mid_dim * 2, 1, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(mid_dim, dim, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.to_q(x).flatten(2).transpose(-2, -1)
        k = self.to_k(x).flatten(2)
        
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        vg = self.to_v_gate(x)
        v, gate = vg.chunk(2, dim=1)
        gate = torch.sigmoid(gate) 
        
        v = v.flatten(2).transpose(-2, -1)
        x_attn = (attn @ v).transpose(-2, -1).reshape(B, -1, H, W)
        
        out = x_attn * gate
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class HighResSHMABlock(nn.Module):
    def __init__(self, dim, window_size=0, num_chunks=1, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_chunks = num_chunks
        
        attn_dim = dim // num_chunks if num_chunks > 1 else dim
        self.attn = EfficientSHMA(attn_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma = nn.Parameter(1e-6 * torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        B, C, H, W = x.shape
        
        if self.window_size > 0:
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))
                _, _, H, W = x.shape

            x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.window_size, self.window_size)

            if self.num_chunks > 1:
                x = x.view(x.shape[0], self.num_chunks, C // self.num_chunks, self.window_size, self.window_size)
                x = x.flatten(0, 1) 

        x = self.attn(x)

        if self.window_size > 0:
            if self.num_chunks > 1:
                x = x.view(-1, self.num_chunks, C // self.num_chunks, self.window_size, self.window_size)
                x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, self.window_size, self.window_size)
            
            B_ = B
            x = x.view(B_, H // self.window_size, W // self.window_size, C, self.window_size, self.window_size)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B_, C, H, W)
            
            if pad_h > 0 or pad_w > 0:
                x = x[:, :, :shortcut.shape[2], :shortcut.shape[3]]

        return shortcut + self.drop_path(self.gamma * x)


class IR_SHMABlock(nn.Module):
    def __init__(self, dim, window_size=16, num_chunks=1, drop_path=0., mlp_ratio=2.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        
        self.attn_block = HighResSHMABlock(dim, window_size, num_chunks, drop_path=0.) 
        
        self.norm2 = nn.LayerNorm(dim)
        
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), int(dim * mlp_ratio), 3, padding=1, groups=int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.gamma1 = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x_norm = x.permute(0, 2, 3, 1) 
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)
        
        x = self.attn_block(x_norm) 
        x = shortcut + self.drop_path(self.gamma1.view(1, -1, 1, 1) * x)
        
        shortcut = x
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)
        
        x = self.conv_ffn(x_norm)
        x = shortcut + self.drop_path(self.gamma2.view(1, -1, 1, 1) * x)
        
        return x

class InstanceEnhancementBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(InstanceEnhancementBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.num_features = num_features
        
        if affine:
            self.gamma_hat = nn.Parameter(torch.zeros(1, num_features, 1, 1))
            self.beta_hat = nn.Parameter(torch.ones(1, num_features, 1, 1) * -1.0)
        else:
            self.register_buffer('gamma_hat', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('beta_hat', torch.ones(1, num_features, 1, 1) * -1.0)
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bn_out = self.bn(x)
        if not self.bn.affine:
            return bn_out
        gamma = self.bn.weight.view(1, -1, 1, 1)
        beta = self.bn.bias.view(1, -1, 1, 1)
        m = self.avg_pool(x)
        delta = self.sigmoid(m * self.gamma_hat + self.beta_hat)
        out = (bn_out - beta) * delta + beta
        
        return out

class SRMLayer(nn.Module):
    def __init__(self, channels):
        super(SRMLayer, self).__init__()
        self.cfc = nn.Conv1d(channels, channels, kernel_size=2, groups=channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        mu = x.mean(dim=(2, 3), keepdim=True).view(b, c, 1)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False).view(b, c, 1)
        
        style = torch.cat([mu, std], dim=2)
        
        z = self.cfc(style)
        z = self.bn(z)
        g = self.sigmoid(z)
        
        return x * g.view(b, c, 1, 1)

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, reduction=16, bias=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(1, num_feat // reduction)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_feat, mid_channels, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_feat, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self, num_feat, ca_reduction=16, bias=True):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias),
            ChannelAttention(num_feat, reduction=ca_reduction, bias=bias)
        )

    def forward(self, x):
        return x + self.body(x)

class ResGFMBlock(nn.Module):
    def __init__(self, num_feat, inter_ratio=2, bias=True):
        super(ResGFMBlock, self).__init__()
        inter_channels = max(1, int(num_feat / inter_ratio))
        self.cond_net = nn.Sequential(
            nn.Conv2d(num_feat, inter_channels, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inter_channels, num_feat * 2, 1, bias=bias) 
        )
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias)

    def forward(self, x):
        cond = self.cond_net(x)
        scale, shift = torch.chunk(cond, 2, dim=1)
        out = self.conv1(x)
        out = out * (scale + 1) + shift
        out = self.act(out)
        out = self.conv2(out)
        return x + out