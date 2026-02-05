Many thanks to the competition host and Kaggle for another engaging challenge - and congratulations to all the participants!
As always, I had a great time learning throughout the competition, and specially it was indeed a crazy race with the deadline for me.Many emotions till the end, now I'm really happy to share a few thoughts here.

## The overall pipeline

![figure of overall pipeline containing 3 stages: orientation correction, heatmap-based keypoints estimation, heatmap-based lead waveform prediction](./figures/overall_pipeline.png)



## Heatmap-based keypoints estimation
An 2D UNet model was trained to predict 57 "feature-rich" keypoints and `43*55=2365` grid keypoints, as show in the figure below

![2422 target keypoints drawed on a reference image of type 0001](./figures/standard_reference_keypoints.png)
I get all 2422 keypoint exact coordinates by inspecting the [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) source code.
Where "main" keypoints are heuristicaly selected, typically around the calibration pulses, splitting ticks, lead names which I consider having rich local features.
>>Note that I had ignored some near-border grid keypoints since it could confuse the model. But it cause lots of headache in the subsequence registering stage, and maybe keeping all `44*57` instead of just `42*55` keypoints is a better choice :D


### Very good initial pseudo label
Standing on the giant's shouder: [ROMA-MINIMA](https://github.com/LSXI7/MINIMA) to obtain a very accurate initial pseudo labeled keypoints. Simply match the type 0001 image to each image in the training set.

![MINIMA RoMa for accurate initial pseudo labeled keypoints](./figures/minima_roma_imcui.png)

Note that I don't do matching then estimate Homography matrix, cause it will generate wrong keypoints coordinate in a non-planar scene or local distortion is heavy. Instead, for modern dense matching models (LoFTR, RoMA,..), we can resample/interpolate the predicted warping flow at arbitary coordinates in an image to estimate the sub-pixel level matched keypoints coordinates on the remaining.

You can try more recent SOTA methods on Image Matching very quickly using this awesome demo: https://huggingface.co/spaces/Realcat/image-matching-webui


### Modeling
2D UNet model was trained to predict 58-channels output heatmap:
- First channel: single 2D heatmap, encoding spatial location of all 2365 grid keypoints. For each keypoint, a small unnormalized Gaussian-like heatmap centered on that keypoint is draw, with sigma=2 pixels relative to the standard reference image (type 0001, 1700x2200), adaptively scale based on the current image's scale (relative to the reference).
- Last 57 channel: Each channel encodes the location of a single "main/feature-rich" keypoint, also using a Gaussian heatmap with sigma=2 pixels, similar to the above setting.


![Visualization of a keypoint detection pipeline's augmented train sample](./figures/augmented_keypoint_detection_train_sample.jpg)


The model was trained end-to-end using multi-task losses. Despite the network can be trained using just BCE loss, I used BCE for the first channel, and Channel-Masked JSD for the rest 57 channels. Each of the 57 channels encode only single keypoint Gaussian heatmap, the spatial distribution has 0 peak (when keypoint is outside of image) or 1 peak (unimodal), unlike multiple peaks (multimodal) distribution as the first channel. Thus, the spatial distribution-based loss (JSD, KLDiv, CE) has better inductive-bias/regularization compared to BCE. Using JSD allow for much faster convergence, confirmed in almost every experiment/project which I had finished in the past.

Also, scaling image size is better than model size. We already know that the patern/context is not very hard to predict for a model in this particular task, which I also found MaxVit (hybrid CNN-Transformer) did not outperform a ConvNeXT-small with much limitted receptive field. So, local patern recognition is enough, allow to use a CNN-only architecture which much easier to scale to larger image size. Larger image size is critically important to obtain a find-grained heatmap with less sub-pixel error. Also, what happen if the test image has the ROI is much smaller to the captured image (camera is far from the capture object), then `longest resize + padding` transform will destroy the details, so resolution must be prioritized.

I measured a keypoint version similar to `AP@0.5-0.95` for grid keypoints and `Accuracy@0.5-0.95` for 57 main keypoints to track the best model. Final config used to train 5-folds models: 
- `3x2048x2048` image size, longest resize + padding with bicubic interpolation
- Output heatmap has stride of 1, shape of `58x2048x2048`
- Model:
  + Encoder: ConvNeXT-small ([convnext_small.fb_in22k_ft_in1k_384](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k_384))
  + Decoder: Standard SMP UNet Decoder with 4 blocks of `[384, 256, 128, 64]` channels. *Tried other options such as PixelShuffle-based decoder, but did not outperform the baseline.*
  + MLP segmentation head: `64 -> 128 -> 58` with GELU and LayerNorm
- Heatmap Gaussian sigma is 2 pixels (*tunned*)
- Multi-task Losses (2 losses): BCE (1st channels) + JSE (2nd-58th channels)
- Multi-task weighting: GLS ([Geometric Loss Strategy](https://arxiv.org/pdf/1904.08492)). *GLS is good, not always the best, but almost the first one I will try in a MTL setup :D*
- Heavy Data Augmentation: **Affine**, **Perspective**, **RandomCrop**, GrayScale, **RandomBrightnessContrast**, ColorJitter, Downscale, Blur, Noise, **Dropout (Coarse, Grid, XYMasking)** with carefully design to preserve enough information. RandomCrop, Dropout at the image level might helps to resolve occlusion/partial crop and encourage better global context learning.
- AdamW optimizer with learning rate `1e-4`, Cosine scheduler
- Model EMA with decay=0.999

After heatmap model was trained, I finetune each fold model using an addition loss to archive sub-pixel accuracy on main keypoints prediction: MSELoss on [DSNT](https://arxiv.org/pdf/1801.07372) prediction and groundtruth coordinate of shape `(57, 2)`, results in totally 3 losses.


### Iterative pseudo labeling

## Rectification
After obtaining heatmap from previous stage, the next task is to decode the heatmap into discreate keypoints and register/order them correctly. The following logics were applied in sequential:
- Decode the 57 main keypoints: simply argmax over 2D spatial heatmap for each 2nd-58th output channel, this way we already known the correct keypoint order. *We can use confident score to determine if a keypoint was outside the image region, but it's not trustworthy since indeed the model is not supervised on the "out-of-region" keypoints (channel-masked in JSD loss). Fortunately, subsequent stages are robust enough to handle WRONG prediction of outside image keypoints.*
- For 5-folds models, we got `(5, 57, 2)` decoded main keypoints. Simply flatten to `(285, 2)`, using those "duplicated" ones to estimate Homography Transformation matrix H (strong assumtion that it's an Affine transform) and relative scale from the **standard reference image (type 0001)** to current images. RANSAC is robust to outlier, so wrong prediction/noise from previous stage is filtered.
- NMS threshold (L2 distance) is set to 20 pixels in reference image, adaptively scaled using the estimated relative scale mentioned above to be suitable for this current image -> decode the first channel "grid" heatmap into a list (variate length) of grid keypoints
- Now the only remain thing is how to 1:1 mapping between a list of predicted grid keypoints and 2365 reference grid keypoints. It seem to be easy at first look, but there're many edge cases which would happen in realife and posiblely in private test. A multi-stages matching algorithm is developed which did solve all provided cases in the training set, not sure it's good enough. Hard to describe, I list some key ideas behind it:
  - Using Homography transformation matrix H estimated in previous step, we have a bijection betwen current coordinate space vs the reference coordinate space.
  - Linear Assignment Matching (Hungarian algorithm) using pairwise L2 distance as cost matrix, disable "imposible" matching by a proper gating cost
  - Use a strict threshold, e.g 8 pixels error is allowed. This prevent False Positive matches where a predicted keypoint are wrongly matched to a reference keypoint. If the paper is not planar but curved/creased/wrinkled then H is no longer accurate, so just a fraction of predicted keypoints were matched.
  - Based on high confident matched keypoints, recompute/interpolate nearby reference keypoints using local Homography matrix (esitimated from nearby matches only) computed for **each** keypoint.
  - This happen in a loop until no new matches be found, iteratively match all predicted keypoints and register theme with correct indices. Miss detection will be replaced by an accurate interpolated version using information from just the nearby predicted keypoints only, partially solve the "local distortion" problem.
- After all, for each image, we got an accurate list of 2422 keypoints (2365 grid keypoints + 57 main keypoints) 


![GIF visualization of how registering algorithm work](./figures/keypoints_register_algorithm.gif)
<center>This GIF describe how the keypoints registering algorithm worked step by step</center>


## Lead Cropping

Given original images and 2422 keypoints estimated from previous stage. All images use the same reference template (type 0001), so it's easier to cropout an arbitary region of interest, predefined using coordinates in reference template. Some choice of cropping methods were tested:
1.  Estimate a single Homography matrix mapping from current image to reference image using nearby "main" keypoints
2. Estimate Piecewise Homography matrix mapping each cell (defined by 4 grid corners) from current image to reference image, using `cv.getPerspectiveTransform` locally -> compute flow map -> resample using `cv2.remap` (or `F.grid_sample` or `scipy.ndimage.map_coordinates`)
3. Same idea as (2), but using `scipy.interpolate.RectBivariateSpline`
4. Same as (2), but for each cell, using `cv2.findHomography` to find local Homography on **K=16** nearby keypoints instead of just **K=4** in (2)

![Visualization of cropping method 1, 2, 4](./figures/cropping_methods.png)


(4) perform the best, since it is not global as in (1) but keep "locality" property enough, well handeling local distortion, but also not too strictly local and sensitive to grid keypoints estimation error as in (2).

![Image visualize misalignment using 1 but correctly alignment using (1) or (4)](./figures/warping_alignment_comparision.png)


## Heatmap-based lead waveform estimation 

Given a warped crop of each lead, another UNet was trained to predict a 2D heatmap of lead waveform.
I think the codec (encoding/decoding operation) is important here. For each lead, I cropout the lead image region slightly wider for both left and right to prevent wrong rectification in previous stage destroy the signal needed for prediction. That is, even if the crop is left-shifted or right-shifted by a small amount of pixels, the rendered waveform still fully included in the image, thus can be recovered by a good model.


![Visualization of cropping and heatmap strategy with detail describing each component](./figures/gt_heatmap.png)

As for the heatmap, I render it in a column-independent way. Each column is an unnormalized 1D-Gaussian heatmap with 1 peak (mu) at the groundtruth value, and a std (sigma) value is fixed or adaptively changed based on the waveform itself. So, each column always represent a probability distribution with single peak. This codec scheme is "nearly lossless", i.e very high SNR during encodeing and decoding back (recompute expectation from a probability density function) operation.


### Combine them all

![Dual encoder UNet architecture](./figures/dual_encoder_unet_architecture.png)


Indeed, I had not trained this final architecture before, and just train it once on all data go get single checkpoint and submit by the deadline. The hyperparameters was select based on heuristic and previous experiments, in which I combine all "what should work" into the final trial. Almost previous experiments does not introduce the VGG19 finegrained/high-resolution encoder, but rather based on a simpler baseline:
- Image size `512x512`, GT waveform is resampled to fixed length of 500, GT heatmap has shape `(1, 512, 512)` where the center region `(1, 512, 500)` actually encode the GT waveform
- Rectification using method (2), Piecewise Perspective Transform(`K=4`)
- UNet model with ConvNext-small encoder, a standard SMP UNet decoder which output heatmap of **stride 1**, shape `(1, 512, 512)`
- Column-wise JSD Loss (i.e, `F.softmax(dim=2)` on predicted tensor of shape `NCHW`)

**Some key insights:**
- Warping **interpolation mode** matters, prevent lossing very little details: `cv2.INTER_LANCZOS4` performed the best and was used in almost experiments
- Heatmap Gaussian sigma=2px relative to the reference template image 0001
- Adaptive sigma scale: the rational behind this is there are some part of waveform which is harder to predict than the other, e.g the sharp peak where the magnitude significantly change in a short time, result in a "near straight line" parallel to the mV axis. A simple method was applied which increase sigma value for waveform value where local standard deviation is large, which also show improvement in local CV:
  ```Python
  SIGMA, ADAPTIVE_FACTOR = 2, 0.4
  local_abs_diff = 0.5 * (np.abs(arr - np.r_[arr[0], arr[:-1]] + np.abs(arr - np.r_[arr[1:], arr[-1]])
  # 3-sigma rule: if > 3*sigma, start using scale >= 1
  adaptive_sigma_arr = SIGMA + ADAPTIVE_FACTOR * np.maximum(local_abs_diff - 3 * SIGMA, 0) / 3
  ```

    ![Visualization of adaptive sigma scale](./figures/adaptive_sigma_scale.png)

- Column-wise JSD loss was used. *In short `JSD` > `CE` >> `BCE`*
- UNet Decoder: Final model uses 6 UNet Decoder blocks, decoder channels `[256,192,160,128,96,64]` corresponding to stride `64 -> 1` with LayerNorm and GELU activation. *Performance better scale with number of parameters. Higher number of channels in high resolution feature map is needed to preserve finegrained texture details, but also increase memory heavily. For upscale type, PixelShuffle-based Decoder was tried but didn't outperform traditional F.interpolate() one. Deformable Convolution (v2 or v4) also be tried as dropin replacement for traditional nn.Conv2d and show better performance, but not used due to slower runtime and argue that gains comes from more parameters count instead.*
- Resolution matter: the use of **input image size of 1024** is critical to keep texture details, bring significant gain over 512. *Before, I had tested if the gain comes from higher input resolution, or higher output resolution by sweep over some modeling configs:*
  + *Image size 512, output heatmap size 1024 (stride 0.5 with an additional x2 upscale UNet Decoder block)*
  + *Image size 512, change encoder stride from 4 to 1 or 2 (modifying the first stem convolution stride)*
  + *Image size 1024, output heatmap size 512*
  + *(much better) Image size 1024, output heatmap size 1024*

- Main encoder: CoAT and ConvNext-large was used. The two kind of architecture show different characteristics. CoAT tend to be slightly better on noisy and occlusion image types, posibly due to larger receptive field and more input-dynamic, hence can use nearby information to guess what was under occlusion. While ConvNext beter at locality and extracting finegrained features, hence better SNR on good and high-resolution images such as phone's photo.

    ![Comparision of Convnext-small 1024 vs CoAT 512 by degradation types](./figures/convnextsmall1024_coat512_comparision_by_image_type.png)
*Comparision is unfair due to different image size (1024 vs 512), but still show some characteristic of each architecture: pure-CNN vs Hybrid CNN-Transformer*

- The last change: VGG19 encoder to extract feature map at stride 1/2/4. We known that this task is strongly benefit from low-level feature map and high-resolution, and VGG is one of the very few architecture which output stride 1 feature map by default. VGG is also used in [RoMA](https://arxiv.org/abs/2305.15404) and proved to be better than ResNet-like architecture in extracting finegrained local feature. I used [vgg19.tv_in1k](https://huggingface.co/timm/vgg19.tv_in1k) which does not use BatchNorm, inspired by Image Super Resolution literature ([EDSR](https://arxiv.org/abs/1707.02921)) and also did not want to messup with BN on small batch size setting.
- The blank template: I use [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) to render an empty image without any lead waveform, act as a blank template with just grids, calibration pulses, separation ticks, lead names. Each lead crop was concatenate with the coresponding grayscale blank template, result in a 4-channels image to be passed to 2D UNet model, instead of original 3-channels RGB image. I hypothesis the template is useful for model to better learning local correlation between rectified image and standard grid template, then internally learning to align accordingly. It also reduce the complexity of learning lead-specific grid layout, hence faster convergence.

    ![Example crop template](./figures/grayscale_template.png)

- Augmentation: The key augmentation was to add small amount of noise (following a truncated normal distribution with sigma=0.4px) into the detected grid keypoints, before the lead cropping procedure (using local Piecewise Homography transform (2)). This mimic the real life type of error while the keypoint detector's prediction is not perfectly accurate. No much gain from usual augmentation like ColorJitter, BrightnessContrast, Grayscale, very small Affine/Perspective transform,.. so I set these augmentation probability to a small value p=0.1
- Training: AdamW optimizer, Cosine LR scheduler, gradient clipping by norm of 1.0 and models are trained for about 70K steps with effective batch size of 8 (batch size 2, gradient accumulation 4)
- Model EMA with decay=0.999

## Image orientation correction
For each image, we can get exact rotation angle relative to the standard reference image using Homography H. I simply train a [efficientvit_b2.r224_in1k](https://huggingface.co/timm/efficientvit_b2.r224_in1k) to jointly predict one of 4 posible rotation 0/90/180/270 degree (classification task) and exact rotation angle encoded by sine/cosine (regression task). During training, heavy augmentation was applied to ensure trained model will be robust on unseen private test. Of couse, the training task is just too easy so that validation accuracy is 100% and angle MAE is just around 1.1 degree.


## Final submission
I write inference code and submit it near the deadline, everything was a mess on that last day.
All submission include inference pipeline for single Image Rotation/Orientation model, 5-folds keypoints detection model.
The first 4 submissions are all estimate lead waveform using single model without ensemble, and prediction dynamic also limitted in range `[-3.2, 3.2]` due to the nature of the heatmap codec. Interestingly, just scale the image size did not work, my model did not generalize well to the new input size, i.e training on input size `[512, 512]` which can encode `[-3.2, 3.2]` waveform then inference on input size `[1024, 512]` which can encode `[-6.4, 6.4]` waveform resulted in very bad SNR. 4 single models were submitted:
- (1) Dual Encoder CoaT Lite Medium + VGG19 on image size `[1024, 1024]`, output heatmap of size `[1024, 1024]` (*first time training, no validation*)
- (2) Dual Encoder ConvNeXT Large + VGG19 on image size `[1024, 1024]`, output heatmap of size `[1024, 1024]` (*first time training, no validation*)
- (3) Single Encoder CoaT Lite Medium on image size `[512, 512]`, output heatmap of size `[512, 512]` (*best learning rate is known*)
- (4) Single Encoder ConvNeXT small on image size `[1024, 1024]`, output heatmap of size `[1024, 1024]` (*best learning rate is known*)

The final submission:
- Ensemble of (1) and (2) with corresponding weight of 0.7-0.3
- Lead II first quarter of 2.5 seconds fusion with weight 0.5-0.5
- Luckily, a single "TALL" model (single encoder ConvNeXT-small) accept input size of `[1024, 512]`, able to handle waveform in range [-6.4, 6.4] was trained and finished in time. It was used to solve the limitted range of main models, act as a refinement stage where first stage's predictions is near the limitation, e.g `np.abs(prediction_signal)` close to 3.2.
- It succesfully scored 22.93 on LB and 22.63 in PB 8 minutes before the competition deadline, wow!

## Source code
- Training code: **update soon**
- Inference notebook: https://www.kaggle.com/code/dangnh0611/5th-place-solution (*I know it look like a mess, and will clean it soon :D*)


Thanks for your attention !
