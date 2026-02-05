import numpy as np
import torch
from ecg.utils import constants
from yagm.transforms.onebumentations.functional import interp_1d
from ecg.utils.interpolate import interp_1d_cv2


class BaseCodec:

    def encode(self, signal):
        raise NotImplementedError

    def batch_decode(self, outputs):
        raise NotImplementedError


class ColumnGaussianHeatmapCodec(BaseCodec):
    def __init__(
        self,
        sigma=2,
        crop_aspect_ratio=1.0,
        H=512,
        W=512,
        L=500,
        cutoff_thres=None,
        adaptive_sigma_scale=None,
        subpixel=False,
        dtype=torch.float32,
        ref_signal_pixels=constants.REF_SIGNAL_PIXELS,
        ref_y_unit_pixels=constants.REF_Y_UNIT_PIXELS,
    ):
        # cutoff_thres follow 3-sigma rule for 1D normal distribution: 0.011108996538242308
        # re-compute sigma (std) relative to gt image (from original sigma which is relative to the reference image of 1700x2200)
        self.gt_sigma = sigma * (L / ref_signal_pixels) / crop_aspect_ratio
        self.crop_aspect_ratio = crop_aspect_ratio
        self.H = H
        self.W = W
        self.L = L
        self.dtype = dtype
        self.cutoff_thres = cutoff_thres
        self.subpixel = subpixel
        self.adaptive_sigma_scale = adaptive_sigma_scale
        self.pad_left = (W - L) // 2
        self.pad_right = W - L - self.pad_left
        assert self.pad_left == self.pad_right
        self.signal_scale_factor = (
            ref_y_unit_pixels * (L / ref_signal_pixels) / crop_aspect_ratio
        )
        crop_h = crop_aspect_ratio * ref_signal_pixels
        self.max_abs_signal = crop_h / 2 / constants.REF_Y_UNIT_PIXELS
        # (H, L)
        self.grid = (
            torch.linspace(0.5, H - 0.5, H, dtype=dtype).unsqueeze(1).expand(-1, L)
        )

    def encode(self, signal):
        # scale the signal
        # range [0, H]
        signal_pixels = -signal * self.signal_scale_factor + self.H / 2
        signal_pixels = torch.from_numpy(signal_pixels).to(self.dtype)
        # range [0, 1], 0 is top (positive), 1 is bottom (negative)
        regress_signal_gt = signal_pixels / self.H

        # compute adaptive sigma
        if self.adaptive_sigma_scale is not None and self.adaptive_sigma_scale > 0.0:
            arr = signal_pixels
            local_abs_diff = (
                np.abs(arr - np.r_[arr[0], arr[:-1]])
                + np.abs(arr - np.r_[arr[1:], arr[-1]])
            ) / 2
            # 3-sigma rule: if > 3*sigma, start using scale >= 1
            gt_sigma = (
                self.gt_sigma
                + self.adaptive_sigma_scale
                * np.maximum(local_abs_diff - 3 * self.gt_sigma, 0)
                / 3
            )
        else:
            gt_sigma = self.gt_sigma

        # generate gaussian-like heatmap
        heatmap = torch.zeros((self.H, self.W), dtype=self.dtype)
        # (H, L) - (1, L)
        roi_heatmap = torch.exp(
            -0.5 * (self.grid - signal_pixels.unsqueeze(0)) ** 2 / gt_sigma**2
        )
        if self.cutoff_thres is not None:
            roi_heatmap[roi_heatmap < self.cutoff_thres] = 0.0

        # refine to obtain sub-pixel accuracy
        if self.subpixel:
            cur_expectation = torch.sum(self.grid * roi_heatmap, axis=0) / torch.sum(
                roi_heatmap, axis=0
            )
            diff = cur_expectation - signal_pixels
            # negative vs positive offset can cause under/over flow when trying to mul/add
            # seem not to be easy, implement it later
            raise NotImplementedError

        heatmap[:, self.pad_left : -self.pad_right] = roi_heatmap
        return heatmap, regress_signal_gt

    def batch_decode(
        self, heatmap: torch.Tensor, offset: torch.Tensor, regress: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            regress: decoded/combined signal in range [0, 1] where 0/1 is top/bottom line
        """
        del offset
        # decode the heatmap (discreate pixel indices)
        B, C, H, W = heatmap.shape
        assert C == 1 and H == self.H and W == self.W
        roi_heatmap = heatmap[:, 0, :, self.pad_left : -self.pad_right]
        amax = torch.argmax(roi_heatmap, dim=1)
        assert amax.shape == (B, self.L)
        # pixel indices -> contigous coordinate
        amax = amax.float() + 0.5

        # decode to signal value
        heatmap_signal = (self.H / 2 - amax) / self.signal_scale_factor
        heatmap_signal = heatmap_signal.cpu()

        if regress is not None:
            if regress.shape[2] == self.W:
                regress = regress[:, 0, self.pad_left : -self.pad_right]
            else:
                regress = regress[:, 0]
            assert regress.shape[1] == self.L
            # [0, 1] -> [1, -1]
            regress_signal = (1 - 2 * regress) * self.max_abs_signal
            regress_signal = regress_signal.cpu()
        else:
            regress_signal = None

        return heatmap_signal, regress_signal
