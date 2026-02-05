Many thanks to the competition host and Kaggle for another engaging challenge—and congratulations to all the participants!

As always, I had a great time learning throughout the competition. It was indeed a crazy race to the deadline for me, filled with many emotions until the very end. I am really happy to share a few thoughts on my solution here.

## The overall pipeline

![figure of overall pipeline containing 3 stages: orientation correction, heatmap-based keypoints estimation, heatmap-based lead waveform prediction](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2Fec58de2d4023acaebcebccc483cdc7f5%2Foverall_pipeline_jpeg_reduce.jpeg?generation=1769461236267800&alt=media)



## Heatmap-based keypoints estimation
A 2D UNet model was trained to predict 57 "feature-rich" keypoints and `43*55=2365` grid keypoints, as shown in the figure below.

![2422 target keypoints drawed on a reference image of type 0001](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2F797872e9a8bc3a4ec6694c1febd27250%2Fstandard_reference_keypoints.png?generation=1769461312813853&alt=media)


I obtained the exact coordinates for all 2,422 keypoints by inspecting the [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) source code. The "main" keypoints were heuristically selected, typically around the calibration pulses, splitting ticks, and lead names, which I consider to have rich local features.

> **Note:** I ignored some near-border grid keypoints since they could confuse the model. However, this caused many headaches in the subsequent registering stage. Perhaps keeping all `44*57` instead of just `42*55` keypoints would have been a better choice :D

### Very good initial pseudo label
I use one of the SOTA opensource dense matching model [MINIMA-RoMa](https://github.com/LSXI7/MINIMA) to obtain very accurate initial pseudo-labeled keypoints. This involved simply matching the type 0001 image to each image in the training set.

![MINIMA RoMa for accurate initial pseudo labeled keypoints](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2Ff0e4ce91d79f784d7968588accb7e7aa%2Fminima_roma_imcui.jpg?generation=1769461426788911&alt=media)


Note that I did not perform matching followed by Homography matrix estimation to warp reference keypoints into current image's space, because it generates wrong keypoint coordinates if the scene is non-planar or if local distortion is heavy. Instead, for modern dense matching models (LoFTR, RoMA, etc.), we can resample/interpolate the predicted warping flow at arbitrary coordinates in an image to estimate the sub-pixel level matched keypoint coordinates on the remaining one.

You can try more recent SOTA methods on Image Matching very quickly using this awesome demo: https://huggingface.co/spaces/Realcat/image-matching-webui


### Modeling
A 2D UNet model was trained to predict a 58-channel output heatmap:
- **First channel:** Single 2D heatmap encoding the spatial location of all 2,365 grid keypoints. For each keypoint, a small unnormalized Gaussian-like heatmap centered on that keypoint is drawn, with `sigma=2 px` relative to the standard reference image (type 0001, `1700x2200`), adaptively scaled based on the current image's scale (relative to the reference).
- **Last 57 channels:** Each channel encodes the location of a single "main/feature-rich" keypoint, also using a Gaussian heatmap with `sigma=2px`, similar to the setting above.

![Visualization of a keypoint detection pipeline's augmented train sample](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2F25fd889ca1f464eb2265f7f7ec155653%2Faugmented_keypoint_detection_train_sample.jpg?generation=1769461526315754&alt=media)
*Visualization of an augmented train samples, left to right: augmented image, visualization of 2nd-58th heatmap channels, first channel encode grid points, overlayed visualization*


The model was trained end-to-end using multi-task losses. Despite the fact that the network can be trained using just BCE loss, I used BCE for the first channel and Channel-Masked JSD (Jensen-Shannon Divergence) for the remaining 57 channels. Each of the 57 channels encodes only a single keypoint Gaussian heatmap; the spatial distribution has 0 peaks (when the keypoint is outside the image) or 1 peak (unimodal), unlike the multiple peaks (multimodal) distribution of the first channel. Thus, a spatial distribution-based loss (JSD, KLDiv, CE) provides better inductive bias/regularization compared to BCE. Using JSD allows for much faster convergence, which I have confirmed in almost every experiment/project I have finished in the past.

Additionally, scaling the image size proved better than scaling the model size. We already know that the pattern/context is not very hard to predict for a model in this particular task (I found MaxVit, a hybrid CNN-Transformer, did not outperform a ConvNeXT-small with a much more limited receptive field). Therefore, local pattern recognition is sufficient, allowing the use of a CNN-only architecture which is much easier to scale to larger image sizes. Larger image size is critically important to obtain a fine-grained heatmap with less sub-pixel error. Furthermore, if the ROI in the test image is much smaller than the captured image (e.g., the camera is far from the object), a `longest resize + padding` transform will destroy details, so resolution must be prioritized.

I measured a keypoint metric similar to `AP@0.5-0.95` for grid keypoints and `Accuracy@0.5-0.95` for the 57 main keypoints to track the best model. The final config used to train the 5-fold models was:
- `3x2048x2048` image size, longest resize + padding with bicubic interpolation.
- Output heatmap has a stride of 1, shape of `58x2048x2048`.
- **Model:**
  - Encoder: ConvNeXT-small ([convnext_small.fb_in22k_ft_in1k_384](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k_384))
  - Decoder: Standard SMP UNet Decoder with 4 blocks of `[384, 256, 128, 64]` channels. *Tried other options such as PixelShuffle-based decoder, but they did not outperform the baseline.*
  - MLP segmentation head: `64 -> 128 -> 58` with GELU and LayerNorm.
- Heatmap Gaussian sigma is 2 pixels (*tuned*).
- **Multi-task Losses (2 losses):** BCE (1st channel) + JSD (2nd-58th channels).
- **Multi-task weighting:** GLS ([Geometric Loss Strategy](https://arxiv.org/pdf/1904.08492)). *GLS is good—not always the best—but almost the first one I will try in a MTL setup :D*
- **Heavy Data Augmentation:** **Affine**, **Perspective**, **RandomCrop**, GrayScale, **RandomBrightnessContrast**, ColorJitter, Downscale, Blur, Noise, **Dropout (Coarse, Grid, XYMasking)** carefully designed to preserve enough information. RandomCrop and Dropout at the image level might help resolve occlusion/partial crops and encourage better global context learning.
- AdamW optimizer with learning rate `1e-4`, Cosine scheduler.
- Model EMA with decay=0.999.

After the heatmap model was trained, I finetuned each fold model using an additional loss to achieve sub-pixel accuracy on main keypoint predictions: MSELoss on [DSNT](https://arxiv.org/pdf/1801.07372) prediction and groundtruth coordinates of shape `(57, 2)`, resulting in 3 total losses.


### Iterative pseudo labeling
I train 5 models on 5 folds to obtain the OOF predictions, decode, then some postprocessing logics defined in the subsequent section [Keypoint Registration](#keypoint-registration) was applied. This process is treated as a denoising process, where I hope model will learn the average/correct truth and skipping the small amount of noises in the initial pseudo label by MINIMA-RoMa. After 1 round, prediction is good enough and this round 1 pseudo label was used to train final keypoint estimation models for submission.


## Keypoint Registration
After obtaining the heatmap from the previous stage, the next task is to decode the heatmap into discrete keypoints and register/order them correctly. The following logic was applied sequentially:
- **Decode the 57 main keypoints:** Simply `argmax` over the 2D spatial heatmap for each of the 2nd-58th output channels. This way, we already know the correct keypoint order. *We can use a confidence score to determine if a keypoint is outside the image region, but it's not trustworthy since the model is not supervised on "out-of-region" keypoints (channel-masked in JSD loss). Fortunately, subsequent stages are robust enough to handle WRONG predictions of outside-image keypoints.*
- For the 5-fold models, we got `(5, 57, 2)` decoded main keypoints. Simply flatten to `(285, 2)`, using those "nearly duplicated" keypoints to estimate the Homography Transformation matrix H (strong assumption that it's an Affine transform) and the relative scale from the **standard reference image (type 0001)** to the current images. RANSAC is robust to outliers, so wrong predictions/noise from the previous stage are filtered.
- NMS threshold (L2 distance) is set to 20 pixels in the reference image, adaptively scaled using the estimated relative scale mentioned above to be suitable for the current image -> decode the first channel "grid" heatmap into a list (variable length) of grid keypoints.
- Now the only remaining task is a 1:1 mapping between the list of predicted grid keypoints and the 2365 reference grid keypoints. It seems easy at first glance, but there are many edge cases that happen in real life (and possibly in the private test set). A multi-stage matching algorithm was developed which solved all provided cases in the training set, though I pretty sure it's not perfect. It would be long to describe fully, but here are some key ideas behind it:
    - Using the Homography transformation matrix H estimated in the previous step, we have a bijection between the current coordinate space and the reference coordinate space.
    - Linear Assignment Matching (Hungarian algorithm) using pairwise L2 distance as the cost matrix, disabling "impossible" matching via a proper gating cost.
    - Use a strict threshold, e.g., 8 pixels error allowed. This prevents False Positive matches where a predicted keypoint is wrongly matched to a reference keypoint. If the paper is not planar but curved/creased/wrinkled, then H is no longer accurate, so only a fraction of predicted keypoints will be matched.
    - Based on high-confidence matched keypoints, recompute/interpolate nearby reference keypoints using a local Homography matrix (estimated from nearby matches only) computed for **each** keypoint.
    - This happens in a loop until no new matches are found, iteratively matching all predicted keypoints and registering them with correct indices. Missed detections will be replaced by an accurate interpolated version using information from just the nearby predicted keypoints, partially solving the "local distortion" problem.
- In the end, for each image, we obtain an accurate list of 2,422 keypoints (2365 grid keypoints + 57 main keypoints).

![GIF visualization of how registering algorithm work](https://raw.githubusercontent.com/dangnh0611/kaggle_ecg_digitization/main/docs/keypoints_register_algorithm.gif)
*This GIF describes how the keypoints registering algorithm worked step by step*


## Lead Cropping

Given the original images and 2,422 keypoints estimated from the previous stage, we can proceed to cropping. All images use the same reference template (type 0001), so it's easier to crop out an arbitrary region of interest, predefined using coordinates in the reference template. Several cropping methods were tested:
1. Estimate a single Homography matrix mapping from the current image to the reference image using nearby "main" keypoints.
2. Estimate a Piecewise Homography matrix mapping each cell (defined by 4 grid corners) from the current image to the reference image, using `cv.getPerspectiveTransform` locally -> compute flow map -> resample using `cv2.remap` (or `F.grid_sample` or `scipy.ndimage.map_coordinates`).
3. Same idea as (2), but using `scipy.interpolate.RectBivariateSpline`.
4. Same as (2), but for each cell, using `cv2.findHomography` to find a local Homography on **K=16** nearby keypoints instead of just **K=4** as in (2).

![Visualization of cropping method 1, 2, 4](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2F9a70470bf215bdb4c2542f31b669dcea%2Fcropping_methods.png?generation=1769461765265829&alt=media)

Method (4) performed the best, since it is not too global as in (1) but keeps the "locality" property enough to well-handle local distortion, without being too strictly local and sensitive to grid keypoint estimation errors as in (2).

![Image visualize misalignment using 1 but correctly alignment using (1) or (4)](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2F6a70a019c4ea65dd59f48f506daa0cdf%2Fwarping_alignment_comparision.png?generation=1769461816637775&alt=media)
*Misalignment due to local distortion using method (1) - see the sharp peaks, but much better results were obtained using method (4)*


## Heatmap-based lead waveform estimation

Given a warped crop of each lead, another UNet was trained to predict a 2D heatmap of the lead waveform.
I think the encoding scheme (codec) is important here. For each lead, I crop out the lead image region slightly wider on both the left and right to prevent slight rectification errors from the previous stage destroying the signal needed for prediction. That is, even if the crop is left-shifted or right-shifted by a small number of pixels, the rendered waveform is still fully included in the image, thus can be recovered by a good model.

![Visualization of cropping and heatmap strategy with detail describing each component](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2F112c6722c4a4df897b94c17d68f65321%2Fgt_heatmap.png?generation=1769461932661483&alt=media)


As for the heatmap, I render it in a column-independent way. Each column is an unnormalized 1D-Gaussian heatmap with 1 peak (mu) at the groundtruth value, and a std (sigma) value is fixed or adaptively changed based on the waveform itself. So, each column always represents a probability distribution with a single peak. This codec scheme is "nearly lossless", i.e., it maintains a very high SNR during the encoding and decoding back (recomputing expectation from a probability density function) operation.

### Modeling

![Dual encoder UNet architecture](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2Fc9541ad0f89cacec836289daff4773c6%2Fdual_encoder_unet_architecture.png?generation=1769461978304704&alt=media)
*UNet architecture with dual-encoder. A VGG19 encodes finegrained features at stride 1/2/4 while another coarse encoder of either CoaT Lite Medium or ConvNeXT Large aggregate global/nearby information, better handles occlusion or captures long-range dependencies*


Indeed, I had not trained this final architecture before; I just trained it once on all data to get a single checkpoint to submitted just before by the deadline. The hyperparameters were selected based on heuristics and previous experiments, in which I combined everything "that should work" into the final trial. All previous experiments did not introduce the VGG19 fine-grained/high-resolution encoder, but rather relied on a simpler baseline:
- Image size `512x512`, GT waveform is resampled to a fixed length of 500, GT heatmap has shape `(1, 512, 512)` where the center region `(1, 512, 500)` actually encodes the GT waveform.
- Rectification using method (2), Piecewise Perspective Transform (`K=4`).
- UNet model with ConvNext-small encoder, a standard SMP UNet decoder which outputs a heatmap of **stride 1**, shape `(1, 512, 512)`.
- Column-wise JSD Loss (i.e., `F.softmax(dim=2)` on predicted tensor of shape `NCHW`).

**Some key insights:**
- Warping **interpolation mode** matters to prevent losing very fine-grained details: `cv2.INTER_LANCZOS4` performed the best and was used in almost all experiments.
- Heatmap Gaussian sigma is 2px relative to the reference template image 0001.
- Adaptive sigma scale: The rationale behind this is that some parts of the waveform are harder to predict than others, e.g., sharp peaks where the magnitude significantly changes in a short time, resulting in a "near straight line" parallel to the mV axis. A simple method was applied which increases the sigma value for waveform values where the local standard deviation is large. Its effectiveness was validated by an improvement in local CV.
    ```python
  SIGMA, ADAPTIVE_FACTOR = 2, 0.4
  local_abs_diff = 0.5 * (np.abs(arr - np.r_[arr[0], arr[:-1]]) + np.abs(arr - np.r_[arr[1:], arr[-1]]))
  # 3-sigma rule: if > 3*sigma, start using scale >= 1
  adaptive_sigma_arr = SIGMA + ADAPTIVE_FACTOR * np.maximum(local_abs_diff - 3 * SIGMA, 0) / 3
    ```

    ![Visualization of adaptive sigma scale](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2Ff68bf9fa035e934146fc6192ea899dab%2Fadaptive_sigma_scale.png?generation=1769462040222299&alt=media)


* Column-wise JSD loss was used. *In short: `JSD` > `CE` >> `BCE`.*
* UNet Decoder: The final model uses 6 UNet Decoder blocks, decoder channels `[256,192,160,128,96,64]` corresponding to stride `64 -> 1` with LayerNorm and GELU activation. *Performance scales better with the number of parameters. A higher number of channels in the high-resolution feature map is needed to preserve fine-grained texture details, but this also increases memory heavily. For the upscale type, a PixelShuffle-based Decoder was tried but didn't outperform the traditional F.interpolate(). Deformable Convolution (v2 or v4) was also tried as a drop-in replacement for traditional nn.Conv2d and showed better performance, but was not used due to slower runtime; I argued that gains came from the increased parameter count instead.*
* Resolution matters: The use of an **input image size of 1024** is critical to keep texture details, bringing significant gains over 512. *Before this, I tested if the gain came from higher input resolution or higher output resolution by sweeping over some modeling configs:*
  * *Image size 512, output heatmap size 1024 (stride 0.5 with an additional x2 upscale UNet Decoder block)*
  * *Image size 512, change encoder stride from 4 to 1 or 2 (modifying the first stem convolution stride)*
  * *Image size 1024, output heatmap size 512*
  * *(Much better) Image size 1024, output heatmap size 1024*


* Main encoder: CoAT and ConvNext-large were used. The two architectures show different characteristics. CoAT tends to be slightly better on noisy and occluded image types, possibly due to a larger receptive field and more input-dynamic nature, hence it can use nearby information to guess what is under occlusion. Meanwhile, ConvNext is better at locality and extracting fine-grained features, hence better SNR on good and high-resolution images such as phone photos.

    ![Comparision of Convnext-small 1024 vs CoAT 512 by degradation types](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2F368dfb45d6efe2d9472dcb5e5410d4af%2Fconvnextsmall1024_coat512_comparision_by_image_type.png?generation=1769462290019202&alt=media)
    
    *Comparison is unfair due to different image sizes (1024 vs 512), but still shows some characteristics of each architecture: pure-CNN vs Hybrid CNN-Transformer.*

* **The "fine" encoder**: VGG19 encoder to extract feature maps at stride 1/2/4. We know that this task strongly benefits from low-level feature maps and high resolution, and VGG is one of the very few architectures which outputs a stride 1 feature map by default. VGG is also used in [RoMA](https://arxiv.org/abs/2305.15404) and proved to be better than ResNet-like architectures in extracting fine-grained local features. I used [vgg19.tv_in1k](https://huggingface.co/timm/vgg19.tv_in1k) which does not use BatchNorm, inspired by Image Super Resolution literature ([EDSR](https://arxiv.org/abs/1707.02921))
* **The blank template**: I use [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) to render an empty image without any lead waveform, acting as a blank template with just grids, calibration pulses, separation ticks, and lead names. Each lead crop was concatenated with the corresponding grayscale blank template, resulting in a 4-channel image to be passed to the 2D UNet model, instead of the original 3-channel RGB image. I hypothesize the template is useful for the model to better learn the local correlation between the rectified image and the standard grid template (aligned perfectly with groundtruth heatmap), allowing it to internally learn to alignment accordingly. It also reduces the complexity of learning lead-specific grid layouts, hence faster convergence

    ![A corresponding grayscale blank template was concatenated to RGB lead image to obtain 4-channel input image](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10254700%2F4e60aab43be2b204de8b5738dbc24966%2Fgrayscale_template.png?generation=1769466319708029&alt=media)

* Augmentation: The key augmentation was to add a small amount of noise (following a truncated normal distribution with `sigma=0.4px`) into the detected grid keypoints before the lead cropping procedure (using local Piecewise Homography transform (2)). This mimics real-life errors since the keypoint detector's prediction is not perfectly accurate. I found not much gain from usual augmentations like ColorJitter, BrightnessContrast, Grayscale, or very small Affine/Perspective transforms, so I set these augmentation probabilities to a small value p=0.1
* Training: AdamW optimizer, Cosine LR scheduler, gradient clipping by norm of 1.0, and models are trained for about 70K steps with an effective batch size of 8 (batch size 2, gradient accumulation 4)
* Model EMA with decay=0.999


**ABLATION STUDY (update soon)**

## Image orientation correction
There are 69 rotated image in the training set. Not sure how many in test set, and wrong orientation will affect the keypoints detection stage. So I train a simple model to correct/standardize image orientation.

For each image, we can get the exact rotation angle relative to the standard reference image using Homography H. I simply trained a [efficientvit_b2.r224_in1k](https://huggingface.co/timm/efficientvit_b2.r224_in1k) to jointly predict one of 4 possible rotations 0/90/180/270 degrees (classification task) and the exact rotation angle encoded by sine/cosine (regression task). During training, heavy augmentation was applied to ensure the trained model would be robust on the unseen private test set. Of course, the training task is just too easy, so the validation accuracy is 100% and angle MAE is just around 1.1 degrees.


## Final submission

I wrote the inference code and submitted it near the deadline; everything was a mess and aweful on that last day..
All submissions include the inference pipeline for a single Image Rotation/Orientation model and 5-fold keypoint detection models.

The first 4 submissions all estimate lead waveforms using a single model without ensemble, and prediction dynamic was also limited to the range `[-3.2, 3.2]` due to the nature of the heatmap codec. Interestingly, just scaling the image size did not work—my model did not generalize well to the new input size. That is, training on input size `[512, 512]` (which can encode `[-3.2, 3.2]` waveforms) and then inferencing on input size `[1024, 512]` (which can encode `[-6.4, 6.4]` waveforms) resulted in very bad SNR.

4 single models were submitted:

* (1) Dual Encoder CoaT Lite Medium + VGG19 on image size `[1024, 1024]`, output heatmap of size `[1024, 1024]` (*first time training, no validation*).
* (2) Dual Encoder ConvNeXT Large + VGG19 on image size `[1024, 1024]`, output heatmap of size `[1024, 1024]` (*first time training, no validation*).
* (3) Single Encoder CoaT Lite Medium on image size `[512, 512]`, output heatmap of size `[512, 512]` (*best learning rate is known*).
* (4) Single Encoder ConvNeXT small on image size `[1024, 1024]`, output heatmap of size `[1024, 1024]` (*best learning rate is known*).

The final submission:

* Ensemble of (1) and (2) with corresponding weights of 0.7-0.3.
* Lead II first quarter of 2.5 seconds fusion with weights 0.5-0.5.
* Luckily, a single "TALL" model (single encoder ConvNeXT-small) accepting an input size of `[1024, 512]` and able to handle waveforms in the range `[-6.4, 6.4]` was trained and finished in time, but just scored relatively low (`SNR~21.7` on single validation fold) due to limitted tunning and training steps. But it is enough, and was used to solve the limited range of the main models, acting as a refinement stage where the first stage's predictions were near the limitation, e.g., `np.abs(prediction_signal)` close to 3.2.
* It successfully scored 22.93 on LB and 22.63 on PB, finished just 8 minutes before the competition deadline—that's insane..


|                                          **MODEL**                                          | **SNR ON TRAIN SET** |  **Public LB** | **Private LB** |   |
|:-------------------------------------------------------------------------------------------:|:--------------------:|:--------------:|:--------------:|---|
| Dual Encoder CoaT Lite Medium + VGG19, image size 1024, heatmap size 1024                   |       **28.006985**      |  **22.63859**  |  **22.34824**  |   |
| Dual Encoder ConvNeXT Large + VGG19, image size 1024, heatmap size 1024                     |      27.484089       |    22.43810    |    22.15886    |   |
| Single Encoder CoaT Lite Medium, image size 512, heatmap size 512                           |       26.050978      |    21.93992    |    21.80577    |   |
| Single Encoder ConvNeXT small, image size 1024, heatmap size 1024                           |       26.081009      |    22.04173    |    21.78336    |   |
| Ensemble (1) and (2) with weight 0.7-0.3, lead fusion, refinement using TALL model 1024x512 |         _N/A_        | **_22.93061_** | **_22.62929_** |   |


## Source code

* Training code: **update soon**
* Inference notebook: https://www.kaggle.com/code/dangnh0611/5th-place-solution (*I know it looks like a mess, I will clean it soon :D*)


---
Thanks for your attention!
