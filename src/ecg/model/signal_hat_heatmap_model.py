import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint # Import checkpointing
from timm.layers import DropPath
from yagm.layers.common import SegmentationHead2d

logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 1: HAT LAYERS
# ==============================================================================

def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)

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
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HAB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 compress_ratio=3, squeeze_factor=30, conv_scale=0.01, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class OCAB(nn.Module):
    def __init__(self, dim, input_resolution, window_size, overlap_ratio, num_heads,
                 qkv_bias=True, qk_scale=None, mlp_ratio=2, norm_layer=nn.LayerNorm,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def unfolded_window_partition(self, x, window_size, overlap_win_size):
        b, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2)
        
        # --- FIXED PADDING LOGIC for OCAB ---
        overlap = overlap_win_size - window_size
        pad_top = overlap // 2
        pad_bottom = overlap - pad_top
        pad_left = overlap // 2
        pad_right = overlap - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        # ------------------------------------

        stride = window_size
        windows = F.unfold(x, kernel_size=overlap_win_size, stride=stride, padding=0)
        windows = windows.transpose(1, 2)
        windows = windows.view(b, -1, c, overlap_win_size, overlap_win_size)
        windows = windows.permute(0, 1, 3, 4, 2).contiguous()
        return windows

    def forward(self, x, x_size, rpi_oca, attn_mask=None):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        x_windows = self.unfolded_window_partition(x, self.window_size, self.overlap_win_size)
        nw = x_windows.shape[1]
        x_windows = x_windows.view(-1, self.overlap_win_size * self.overlap_win_size, c)
        q_windows = window_partition(x, self.window_size)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)
        
        q = self.qkv(q_windows).reshape(-1, self.window_size*self.window_size, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        k, v = self.qkv(x_windows).reshape(-1, self.overlap_win_size*self.overlap_win_size, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)[1:]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = rpi_oca.view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(nw * b, self.window_size * self.window_size, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(-1, self.window_size, self.window_size, c)
        x = window_reverse(x, self.window_size, h, w)
        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class RHAG(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, overlap_ratio,
                 depth, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 compress_ratio=3, squeeze_factor=30, conv_scale=0.01):
        super(RHAG, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            # Safe drop path extraction
            layer_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            
            if i == depth - 1:
                self.layers.append(
                    OCAB(dim, input_resolution, window_size, overlap_ratio, num_heads,
                         qkv_bias, qk_scale, mlp_ratio, norm_layer,
                         drop, attn_drop, layer_drop_path, nn.GELU)
                )
            else:
                self.layers.append(
                    HAB(dim, input_resolution, num_heads, window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        compress_ratio=compress_ratio, squeeze_factor=squeeze_factor,
                        conv_scale=conv_scale, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=layer_drop_path,
                        norm_layer=norm_layer)
                )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size, params):
        # --- GRADIENT CHECKPOINTING IMPL ---
        def _inner_forward(x):
            shortcut = x
            for layer in self.layers:
                if isinstance(layer, OCAB):
                    x = layer(x, x_size, params['rpi_oca'], params['attn_mask'])
                else:
                    x = layer(x, x_size, params['rpi_sa'], params['attn_mask'])
            x = x.view(x.shape[0], x_size[0], x_size[1], -1).permute(0, 3, 1, 2)
            x = self.conv(x)
            x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
            return x + shortcut

        if self.training:
            return checkpoint.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            return _inner_forward(x)

# ==============================================================================
# SECTION 2: SIGNAL HAT HEATMAP MODEL (With STRIDING)
# ==============================================================================

class ColumnDSNT(nn.Module):
    def __init__(self, height, width, act):
        super().__init__()
        self.act = act
        self._need_col_norm = not isinstance(act, nn.Softmax)
        self.grid = torch.linspace(0.5 / height, 1 - 0.5 / height, height)[
            None, None, :, None
        ].repeat(1, 1, 1, width)

    def forward(self, heatmap):
        B, C, H, W = heatmap.shape
        heatmap = self.act(heatmap)
        if self._need_col_norm:
            prob = heatmap / heatmap.sum(dim=2, keepdim=True)
        else:
            prob = self.act(heatmap)
        y = torch.sum(self.grid.to(prob) * prob, dim=2)
        return y


class SignalHATHeatmapModel(nn.Module):
    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self.cfg = cfg
        self.subpixel_mode = cfg.subpixel_mode
        
        # --- Config Extraction ---
        img_size = getattr(cfg, 'img_size', 512)
        in_chans = getattr(cfg, 'in_chans', 3)
        embed_dim = getattr(cfg, 'embed_dim', 48) # Default to Tiny
        
        depths = getattr(cfg, 'depths', (6, 6, 6, 6))
        num_heads = getattr(cfg, 'num_heads', (6, 6, 6, 6))
        window_size = getattr(cfg, 'window_size', 8)
        
        compress_ratio = getattr(cfg, 'compress_ratio', 3)
        squeeze_factor = getattr(cfg, 'squeeze_factor', 24)
        conv_scale = getattr(cfg, 'conv_scale', 0.01)
        overlap_ratio = getattr(cfg, 'overlap_ratio', 0.5)
        mlp_ratio = getattr(cfg, 'mlp_ratio', 2.0)
        qkv_bias = getattr(cfg, 'qkv_bias', True)
        qk_scale = getattr(cfg, 'qk_scale', None)
        drop_rate = getattr(cfg, 'drop_rate', 0.0)
        attn_drop_rate = getattr(cfg, 'attn_drop_rate', 0.0)
        drop_path_rate = getattr(cfg, 'drop_path_rate', 0.1)
        ape = getattr(cfg, 'ape', False)
        img_range = getattr(cfg, 'img_range', 1.0)
        
        # --- STRIDE FACTOR (Hardcoded to 4 for memory safety) ---
        # 512x512 -> 128x128
        self.stride_factor = 4
        self.img_size = img_size
        self.internal_size = img_size // self.stride_factor
        self.embed_dim = embed_dim
        self.img_range = img_range
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        
        # 1. Pre-process (Mean Shift)
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        
        # 2. Strided Stem (Input -> Internal Res)
        # Replaces simple conv_first with downsampling stack
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1), # Stride 2
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1) # Stride 2 (Total 4)
        )

        # 3. Deep Feature Extraction (Body at 128x128)
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        if getattr(cfg, 'ape', False):
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.internal_size**2, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            self.ape = True
        else:
            self.ape = False
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        for i_layer in range(len(depths)):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(self.internal_size, self.internal_size),
                num_heads=num_heads[i_layer],
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Relative Position Indices (Calculated on internal_size)
        self.register_buffer('relative_position_index_SA', self._calculate_rpi_sa())
        self.register_buffer('relative_position_index_OCA', self._calculate_rpi_oca())

        # 4. Upsampling Neck (128x128 -> 512x512)
        # Use bilinear for smooth heatmap
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=self.stride_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.GELU()
        )

        # 5. Segmentation Head
        self.seg_head = SegmentationHead2d(
            in_channels=embed_dim,
            out_channels=cfg.seg_head.out_channels,
            ksize=cfg.seg_head.ksize,
            act=None,
            hidden_channels=cfg.seg_head.hidden_channels,
            hidden_norm=cfg.seg_head.hidden_norm,
            hidden_act=cfg.seg_head.hidden_act,
        )

        # 6. DSNT / Activation
        heatmap_loss_name = global_cfg.loss.heatmap_loss
        if heatmap_loss_name == "bce":
            self.heatmap_act = nn.Sigmoid()
        elif heatmap_loss_name in ["l1", "mse"]:
            self.heatmap_act = nn.Identity()
        elif heatmap_loss_name in ["cce", "kl", "bikl", "jsd"]:
            self.heatmap_act = nn.Softmax(dim=2)
        else:
            raise ValueError

        if self.subpixel_mode == "dsnt":
            self.col_dsnt = ColumnDSNT(
                global_cfg.data.heatmap_hwl[0],
                global_cfg.data.heatmap_hwl[1],
                self.heatmap_act,
            )

        # Init Head
        pos_rates = torch.tensor(5e-3)
        bias_init = -torch.log(1.0 / pos_rates - 1.0)
        nn.init.constant_(self.seg_head[-1].bias, bias_init)
        logger.info("Init bias value of segmentation head to %s", bias_init)

    def _calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def _calculate_rpi_oca(self):
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)
        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_ori_flatten = torch.flatten(coords_ori, 1)
        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_ext_flatten = torch.flatten(coords_ext, 1)
        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        return relative_coords.sum(-1)

    def _calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.window_size // 2), slice(-self.window_size // 2, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.window_size // 2), slice(-self.window_size // 2, None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward_hat_features(self, x):
        # 0. Handle uint8
        if x.dtype == torch.uint8:
             x = x.float() / 255.0

        # 1. Preprocess
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        # 2. Downsample Stem (512 -> 128)
        x = self.stem(x)
        
        # 3. Transformer Body (128x128)
        x_size = (x.shape[2], x.shape[3])
        attn_mask = self._calculate_mask(x_size).to(x.device)
        params = {
            'attn_mask': attn_mask,
            'rpi_sa': self.relative_position_index_SA,
            'rpi_oca': self.relative_position_index_OCA
        }

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        
        # 4. Post-Body Conv
        x = self.conv_after_body(x)
        
        # 5. Upsample Back (128 -> 512)
        x = self.upsample(x)
        return x

    def forward(self, x):
        features = self.forward_hat_features(x)
        heatmap = self.seg_head(features)

        if self.subpixel_mode is None:
            prob_heatmap = heatmap
            offset_heatmap = None
            combined_signal = None
        elif self.subpixel_mode == "dsnt":
            prob_heatmap = heatmap
            offset_heatmap = None
            combined_signal = self.col_dsnt(prob_heatmap)
        elif self.subpixel_mode == "offset":
            raise NotImplementedError
        else:
            raise ValueError

        return prob_heatmap, offset_heatmap, combined_signal