import torch
import torch.nn as nn
import torch.nn.functional as F
from yagm.layers.common import SegmentationHead2d
from timm.layers import DropPath
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# PART 1: Official NAFNet Components
# Ref: https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py
# ==============================================================================


# @TODO - use manual implementation from timm instead to prevent permute() twice
class LayerNorm2d(nn.Module):
    """
    LayerNorm that supports NCHW data format.
    Matches the official 'LayerNorm2d' from NAFNet repo.
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(
            x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """
    Official SimpleGate: Split in half and multiply.
    Replaces GELU/ReLU.
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Official NAFBlock implementation.
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, drop_path=0.0):
        super().__init__()

        # --- Path 1: Spatial Block ---
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )

        # SimpleGate divides channels by 2
        self.sg = SimpleGate()

        # Simplified Channel Attention (SCA)
        # Operates on the gated features (dw_channel // 2)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # --- Path 2: FFN Block ---
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Norms & Drops
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        # Official NAFNet uses DropPath here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale (Initialized to 0)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        input = x

        # Block 1
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)  # Attention without Sigmoid
        x = self.conv3(x)
        x = self.dropout1(x)

        y = input + self.drop_path(x * self.beta)

        # Block 2
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + self.drop_path(x * self.gamma)


# ==============================================================================
# PART 2: NAFNet Backbone (U-Net Structure)
# ==============================================================================


class NAFNetBackbone(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        drop_path_rate=0.0,
        drop_out_rate=0.0,
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Calculate Drop Path Rates (Linear Decay)
        # Total blocks = intro + encs + middle + decs
        total_blocks = sum(enc_blk_nums) + middle_blk_num + sum(dec_blk_nums)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        dpr_idx = 0

        chan = width
        # --- Encoder ---
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            chan,
                            drop_out_rate=drop_out_rate,
                            drop_path=dpr[dpr_idx + i],
                        )
                        for i in range(num)
                    ]
                )
            )
            dpr_idx += num
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))  # Stride 2 Downsample
            chan = chan * 2

        # --- Middle ---
        self.middle_blks = nn.Sequential(
            *[
                NAFBlock(chan, drop_out_rate=drop_out_rate, drop_path=dpr[dpr_idx + i])
                for i in range(middle_blk_num)
            ]
        )
        dpr_idx += middle_blk_num

        # --- Decoder ---
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),  # Sub-pixel upscale
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            chan,
                            drop_out_rate=drop_out_rate,
                            drop_path=dpr[dpr_idx + i],
                        )
                        for i in range(num)
                    ]
                )
            )
            dpr_idx += num

    def forward(self, x):
        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip  # Skip Connection
            x = decoder(x)

        return x


# ==============================================================================
# PART 3: Signal Heatmap Wrapper
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


class SignalNAFHeatmapModel(nn.Module):
    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self.cfg = cfg
        self.subpixel_mode = cfg.subpixel_mode

        # --- Config Extraction ---
        # "width=32" matches NAFNet-Local (approx 1.5M params if shallow)
        # "width=64" matches NAFNet-width64 (SOTA baseline)
        width = getattr(cfg, "width", 32)
        enc_blk_nums = getattr(cfg, "enc_blk_nums", [2, 2, 4, 8])
        dec_blk_nums = getattr(cfg, "dec_blk_nums", [2, 2, 2, 2])
        middle_blk_num = getattr(cfg, "middle_blk_num", 12)
        drop_path_rate = getattr(cfg, "drop_path_rate", 0.0)
        drop_out_rate = getattr(cfg, "drop_out_rate", 0.0)
        img_channel = getattr(cfg, "in_chans", 3)

        self.backbone = NAFNetBackbone(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
            drop_path_rate=drop_path_rate,
            drop_out_rate=drop_out_rate,
        )

        # --- Segmentation Head ---
        # NAFNet backbone outputs 'width' channels
        self.seg_head = SegmentationHead2d(
            in_channels=width,
            out_channels=cfg.seg_head.out_channels,
            ksize=cfg.seg_head.ksize,
            act=None,
            hidden_channels=cfg.seg_head.hidden_channels,
            hidden_norm=cfg.seg_head.hidden_norm,
            hidden_act=cfg.seg_head.hidden_act,
        )

        # --- DSNT / Output Activation ---
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

        # Init Head Bias
        pos_rates = torch.tensor(5e-3)
        bias_init = -torch.log(1.0 / pos_rates - 1.0)
        nn.init.constant_(self.seg_head[-1].bias, bias_init)

        # --- 4. PRETRAINED LOADING LOGIC ---
        # Automatically load if path is provided in config
        pretrained_path = getattr(cfg, "pretrained", None)
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, path):
        """
        Robust loader for NAFNet weights.
        Prevents silent failures by enforcing that ALL backbone keys must be present.
        """
        logger.info(f"Context: Loading pretrained weights from {path}")

        # 1. Load Checkpoint
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except FileNotFoundError:
            raise FileNotFoundError(f"Pretrained weights not found at: {path}")

        if "params" in checkpoint:
            state_dict = checkpoint["params"]
        else:
            state_dict = checkpoint

        # 2. Remap Keys (Official -> Backbone)
        new_state_dict = {}
        backbone_prefixes = [
            "intro",
            "encoders",
            "middle_blks",
            "decoders",
            "ups",
            "downs",
        ]

        for k, v in state_dict.items():
            if any(k.startswith(p + ".") for p in backbone_prefixes):
                new_key = f"backbone.{k}"
                new_state_dict[new_key] = v

        # 3. SAFETY CHECK: Verify Backbone Integrity
        # We want to ensure that EVERY key in self.backbone exists in the loaded file.
        # This catches mismatching 'width', 'depths', or 'enc_blk_nums'.

        model_state_dict = self.state_dict()
        # Filter to get only backbone keys from the current initialized model
        expected_backbone_keys = {
            k for k in model_state_dict.keys() if k.startswith("backbone.")
        }
        loaded_backbone_keys = set(new_state_dict.keys())

        # Find keys that the model expects but the checkpoint doesn't have
        missing_backbone_keys = expected_backbone_keys - loaded_backbone_keys

        if len(missing_backbone_keys) > 0:
            # Format error message nicely
            missing_example = list(missing_backbone_keys)[:3]
            raise RuntimeError(
                f"CRITICAL PRETRAIN MISMATCH: The provided checkpoint is missing {len(missing_backbone_keys)} "
                f"backbone keys required by your current config.\n"
                f"Examples of missing keys: {missing_example}\n"
                f"Possible causes:\n"
                f"1. You are loading 'width32' weights into a 'width64' config (or vice versa).\n"
                f"2. Your 'enc_blk_nums' or 'middle_blk_num' do not match the checkpoint structure."
            )

        # 4. Load with shape checking
        # We use strict=False to ignore 'seg_head' (which we expect to be missing),
        # but since we manually verified the backbone above, we are safe.
        try:
            msg = self.load_state_dict(new_state_dict, strict=False)
            logger.info(
                f"Pretrained backbone loaded successfully. (Ignored {len(msg.unexpected_keys)} keys from original head)"
            )
        except RuntimeError as e:
            # Catch Shape Mismatches (e.g. loading 64-channel weight into 32-channel layer)
            raise RuntimeError(
                f"SHAPE MISMATCH: The config dimensions do not match the pretrained weights.\nOriginal Error: {str(e)}"
            )

    def forward(self, x):
        # 0. Handle uint8 [0, 255] input if needed
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # 1. NAFNet Backbone
        features = self.backbone(x)

        # 2. Heatmap Head
        heatmap = self.seg_head(features)

        # 3. DSNT / Output
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
