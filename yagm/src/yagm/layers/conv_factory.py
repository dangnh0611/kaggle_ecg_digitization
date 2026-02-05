# @TODO - proper weight init for all modules

import torch
import torch.nn as nn
import torchvision.ops
from torch.nn.init import eye_, zeros_


class DCNv2Wrapper(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super(DCNv2Wrapper, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # --- 1. Offset and Mask Generation ---
        # v2 requires both offsets and a modulation mask.
        # We need (2 * k * k) output channels for offsets (x, y)
        # We need (1 * k * k) output channels for mask (scalar weight)
        # Total = 3 * k * k
        self.num_offset_mask = 3 * self.kernel_size[0] * self.kernel_size[1]

        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            self.num_offset_mask,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        # --- 2. The Actual Convolution ---
        # Note: We use the functional API directly here because the class wrapper
        # in some older torchvision versions didn't expose the mask argument clearly.
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        # 1. Initialize weights like a normal Conv2d
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        # 2. Initialize offset and mask generators
        # Constant 0 for offsets (so it starts as standard conv)
        # Constant 0.5 or similar for mask (so it starts unmodulated? No, usually 0 is fine,
        # but standard practice is 0 for offsets, and sigmoid(0)=0.5 for mask).
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)

    def forward(self, x):
        # 1. Generate Offsets and Mask
        out = self.offset_mask_conv(x)

        # Split the output into offset and mask
        # chunk(2) won't work because offset is 2/3 and mask is 1/3 of the channels
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)

        # 2. Apply Sigmoid to Mask
        # The modulation mask must be in range [0, 1]
        mask = torch.sigmoid(mask)

        # 3. Apply Modulated Deformable Conv
        return torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class DCNv4Wrapper(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        **kwargs,  # Captures DCNv4 specific args (e.g., offset_scale, dw_kernel_size)
    ):
        """
        Wrapper class which 'seem' to be compatible with nn.Conv2d.
        Ref: https://github.com/OpenGVLab/DCNv4/blob/main/DCNv4_op/DCNv4/modules/dcnv4.py

        Args:
            kwargs: default to
                offset_scale=1.0
                dw_kernel_size=None
                center_feature_scale=False
                remove_center=False
                output_bias=True
                without_pointwise=False
        """
        super(DCNv4Wrapper, self).__init__()

        # --- 1. Argument Standardization ---
        # DCNv4 implementation expects integers, while Conv2d allows tuples.
        # We assume square kernels if tuples are provided.
        def _to_int(param):
            if isinstance(param, (list, tuple)):
                assert all([e == param[0] for e in param])
                return param[0]
            else:
                return param

        k = _to_int(kernel_size)
        s = _to_int(stride)
        p = _to_int(padding)
        d = _to_int(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = s
        self.padding = p
        self.dilation = d
        self.kernel_size = k

        # --- 2. Channel Mismatch Handling ---
        # DCNv4 typically acts as a block where In_Channels == Out_Channels.
        # If they differ, we need an adapter (1x1 Conv).
        self.adapter_needed = in_channels != out_channels

        # If we need an adapter, we disable bias in DCN and add it to the adapter instead
        dcn_output_bias = bias and not self.adapter_needed

        # --- 3. Initialize Core DCNv4 ---
        from DCNv4 import DCNv4

        self.dcn = DCNv4(
            channels=in_channels,
            kernel_size=k,
            stride=s,
            pad=p,
            dilation=d,
            group=groups,
            output_bias=dcn_output_bias,
            **kwargs,
        )

        # --- 4. Initialize Adapter (if needed) ---
        if self.adapter_needed:
            self.adapter = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
                device=device,
                dtype=dtype,
            )
        else:
            self.adapter = None

        # Handle Device/Dtype placement
        if device or dtype:
            self.to(device=device, dtype=dtype)

    def forward(self, x):
        # Input x: (N, C, H, W)
        N, C, H, W = x.shape

        # 1. Reshape for DCNv4: (N, C, H, W) -> (N, H*W, C)
        # Permute to channel-last format
        x_in = x.permute(0, 2, 3, 1).reshape(N, H * W, C)

        # 2. Apply DCNv4
        # We pass (H, W) explicitly so DCN doesn't have to infer from sqrt(L)
        x_out = self.dcn(x_in, shape=(H, W))

        # 3. Calculate Output Spatial Dimensions
        # Standard Conv output size formula
        h_out = (
            H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1
        w_out = (
            W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1

        # 4. Reshape back: (N, L_out, C) -> (N, C, H_out, W_out)
        # Note: DCNv4 output is flattened L, we must reshape carefully
        x_out = x_out.view(N, h_out, w_out, self.in_channels).permute(0, 3, 1, 2)

        # 5. Apply Channel Adapter if necessary
        if self.adapter:
            x_out = self.adapter(x_out)

        return x_out


def replace_conv2d_with_dcnv4(model, replace_pointwise=False, dcn_cls=DCNv4Wrapper):
    """
    Recursively replaces nn.Conv2d layers with DeformableConv2dV4.

    Args:
        model (nn.Module): The model to modify.
        replace_pointwise (bool): If False (default), 1x1 convolutions are NOT replaced.
                                  If True, even 1x1 convolutions are converted to DCNv4.
        dcn_cls (class): The DCN wrapper class to use.

    Usage:
    ```
    # 1. Setup Model
    import torchvision
    model = torchvision.models.resnet50(pretrained=True)
    # 2. Replace spatial convs (3x3), keep pointwise (1x1) as is
    replace_conv2d_with_dcnv4(model, replace_pointwise=False)
    # 3. Model is ready for fine-tuning
    ```
    """

    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # 1. Analyze constraints
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            bias = module.bias is not None

            # Normalize kernel size check (handle tuple vs int)
            k_h = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
            k_w = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size

            # CHECK: Should we replace this layer?
            is_pointwise = k_h == 1 and k_w == 1

            if is_pointwise and not replace_pointwise:
                # Skip 1x1 convs if flag is False
                continue

            # 2. Create the new DCNv4 Layer
            new_layer = dcn_cls(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                without_pointwise=False,
            )

            # 3. SMART WEIGHT LOADING ("Center-Weight" Strategy)

            # A. Initialize Offsets/Masks to Zero (Start as standard grid)
            if hasattr(new_layer.dcn, "offset_mask"):
                zeros_(new_layer.dcn.offset_mask.weight)
                zeros_(new_layer.dcn.offset_mask.bias)

            # B. Transfer Pretrained Weights
            original_weight = module.weight.data  # Shape: (C_out, C_in/g, K, K)

            # Find the center index (e.g., idx 1 for 3x3, idx 0 for 1x1)
            center_h = k_h // 2
            center_w = k_w // 2

            # Extract the weights from the center position
            center_weight_slice = original_weight[:, :, center_h, center_w]

            if new_layer.adapter_needed:
                # Case 1: Channel Mismatch (In != Out)
                # Load weights into the 1x1 adapter wrapper
                new_layer.adapter.weight.data = center_weight_slice.view_as(
                    new_layer.adapter.weight.data
                )

                # Set DCN internal projections to Identity
                eye_(new_layer.dcn.value_proj.weight)
                zeros_(new_layer.dcn.value_proj.bias)
                eye_(new_layer.dcn.output_proj.weight)
                zeros_(new_layer.dcn.output_proj.bias)

                if bias:
                    new_layer.adapter.bias.data = module.bias.data
            else:
                # Case 2: Channel Match (In == Out)
                # Load weights into the DCN output projection

                # Set Value Proj to Identity
                eye_(new_layer.dcn.value_proj.weight)
                zeros_(new_layer.dcn.value_proj.bias)

                # Set Output Proj to the extracted Center Weights
                # Linear layer expects (Out, In), which matches our slice (C_out, C_in)
                new_layer.dcn.output_proj.weight.data = center_weight_slice

                if bias:
                    new_layer.dcn.output_proj.bias.data = module.bias.data

            # 4. Apply Replacement
            setattr(model, name, new_layer)
            print(
                f"Replaced {name}: Conv2d({in_channels}, {out_channels}, k={kernel_size}) -> DCNv4"
            )

        else:
            # Recursively apply to child modules
            replace_conv2d_with_dcnv4(module, replace_pointwise, dcn_cls)


def get_conv_layer(name):
    if name == "conv2d":
        return nn.Conv2d
    elif name == "DCNv2":
        return DCNv2Wrapper
    elif name == "DCNv4":
        return DCNv4Wrapper
    else:
        raise ValueError(f"Unknown convolution type: {name}")


CONV2D_TYPES = (nn.Conv2d, DCNv2Wrapper, DCNv4Wrapper)
