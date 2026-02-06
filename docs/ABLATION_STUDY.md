# Ablation Study

**In this section, we provide some ablation study regarding the design of the 2D UNet for Waveform Estimation**.


- [Ablation Study](#ablation-study)
    - [Important notes](#important-notes)
    - [Backbone choices](#backbone-choices)
    - [Interpolation method](#interpolation-method)
    - [Gaussian heatmap sigma](#gaussian-heatmap-sigma)
    - [Adaptive sigma scale](#adaptive-sigma-scale)
    - [Loss function](#loss-function)
    - [UNet Decoder design](#unet-decoder-design)
    - [UNet Decoder scaling](#unet-decoder-scaling)
    - [Input image \& Output heatmap resolution](#input-image--output-heatmap-resolution)
    - [Grid keypoints coordinate source](#grid-keypoints-coordinate-source)
    - [Random Keypoints Offset Augmentation](#random-keypoints-offset-augmentation)
    - [Blank template as auxiliary input channel](#blank-template-as-auxiliary-input-channel)
    - [Better rectification/warping method](#better-rectificationwarping-method)




### Important notes
**Almost all baseline follow this simple setup (also declared above), unless explicitly specified:**

- Image size 512x512, GT waveform is resampled to a fixed length of 500, GT heatmap has shape (1, 512, 512) where the center region (1, 512, 500) actually encodes the GT waveform.

- Rectification/cropping using method (2), i.e Piecewise Perspective Transform (K=4)

- UNet model with ConvNext-small encoder, a standard SMP UNet decoder which outputs a heatmap of stride 1, shape (1, 512, 512)

- Column-wise JSD Loss, i.e. using F.softmax(dim=2) on predicted tensor of shape NCHW

- Model is trained for 50K steps with Model EMA (decay=0.995)


**All experiments are measured using official competition metrics to compute SNR, with just small change:**

- We also include the short 2.5 second of lead II into evaluation. So the actual duration of lead II is actually 12.5 seconds instead of just 10 seconds.

- In most cases, our  model’s predictions dynamic range is only limited in \[-3.2, 3.2] due to the nature of heatmap codec. We keep ground truth as is and did not clip accordingly to compute the SNR metrics, reflecting the real performance.


### Backbone choices

Benchmarking different image backbones as encoders in our UNet model. All backbones are initialized with Imagenet-pretrained weight. We report the best result for each candidate by just tuning the learning rate suitable for each one. CoaT was superior to others, even with less parameters count. So in the final submission, we select CoaT (Hierarchical Hybrid CNN-Transformer) and ConvNeXT (pure-CNN) to improve modeling diversity.

|                                              |                              |
| :------------------------------------------: | :--------------------------: |
|                 **Backbone**                 | **Validation SNR on fold 0** |
|              ConvNeXT-Small \[4]             |           22.874523          |
|              ConvNeXT-Large \[4]             |           23.014694          |
|             ConvNeXT-XLarge \[4]             |          23.132069           |
|             EfficientNet B5 \[23]            |           22.988173          |
| Flash Intern Image B (using DCNv4) \[18, 24] |           22.951166          |
|               MaxVit-Tiny \[11]              |           22.853954          |
|             CoaT-Lite-Medium \[5]            |         **23.380785**        |
|    NAFNet Width 32, SIDD pretrained \[25]    |           21.825632          |


### Interpolation method

The effect of warping method and interpolation mode. It’s quite clear why LANCZOS4 is superior and used in almost all experiments.

|                                                   |                        |                              |
| :-----------------------------------------------: | :--------------------: | :--------------------------: |
|                 **Warping method**                | **Interpolation mode** | **Validation SNR on fold 0** |
| (1) Global Homography using nearby main keypoints |         lanczos        |           21.545642          |
|      (2) Piecewise Local Homography with K=4      |         lanczos        |         **22.929716**        |
|      (2) Piecewise Local Homography with K=4      |          cubic         |           22.924128          |
|      (2) Piecewise Local Homography with K=4      |          area          |           22.908777          |
|      (2) Piecewise Local Homography with K=4      |         linear         |           22.828297          |
|      (2) Piecewise Local Homography with K=4      |         nearest        |           21.879118          |
|     (3) scipy.interpolate.RectBivariateSpline     |         lanczos        |           22.846357          |
|      (4) Piecewise Local Homography with K=16     |         lanczos        |         **23.086725**        |


### Gaussian heatmap sigma

Effect of Gaussian heatmap sigma. We found sigma\~2 pixels (relative to standard template image of type 0001) worked best empirically.

|                        |                              |
| :--------------------: | :--------------------------: |
| **Heatmap sigma (px)** | **Validation SNR on fold 0** |
|            1           |           22.718933          |
|           1.5          |         **22.916407**        |
|            2           |           22.874523          |
|           2.5          |           22.867516          |
|            3           |           22.764025          |
|            4           |           22.646490          |
|            6           |           22.354689          |


### Adaptive sigma scale

Effect of adaptive sigma scale (with heatmap sigma of 2 pixels). We later choose a sigma scale factor of 0.4 for all final experiments.

|                                 |                              |
| :-----------------------------: | :--------------------------: |
| **Adaptive sigma scale factor** | **Validation SNR on fold 0** |
|          0.0 (disable)          |           22.862576          |
|               0.25              |           22.912252          |
|               0.5               |         **22.929716**        |
|               0.75              |           22.878313          |
|               1.0               |           22.739464          |


### Loss function

Effect of loss function design. Spatial distribution-based loss function (JSD, CE) clearly outperforms other choices. We use JSD for most of our experiments.

|                               |                              |
| :---------------------------: | :--------------------------: |
|       **Loss function**       | **Validation SNR on fold 0** |
|         Columnwise JSD        |         **22.901585**        |
| Columnwise Cross Entropy (CE) |           22.876020          |
|   Binary Cross Entropy (BCE)  |           22.715128          |


### UNet Decoder design

Effect of UNet Decoder design. We found that the modern design of using LayerNorm in combination with GELU activation provides slightly better results. DCNv2 provides quite a large improvement (also additional extra number of parameters), but training speed decreases significantly. We got inspiration from using DCN in our experiment from the Deformable Alignment field, where DCN heavily dominates \[20, 21, 22]. We have a strong assumption that DCN is very suitable for this task, so we can build a much better solution in both accuracy and efficiency based on DCN building blocks. That’s also the reason we tried the Intern-Image variant as an UNet Encoder choice, but did not succeed due to lack of regularization in offset prediction range. Due to lack of time, we left these for future experiments and research.

|                                                                                         |                              |
| :-------------------------------------------------------------------------------------: | :--------------------------: |
|                                 **UNet Decoder design**                                 | **Validation SNR on fold 0** |
|                         Nearest Interpolation + Layernorm + GELU                        |           22.894238          |
|                         Pixelshuffle Upsample + Layernorm + GELU                        |           22.842560          |
| Nearest Interpolation + BatchNorm + ReLU (default config of segmentation-model-pytorch) |           22.864965          |
|           Nearest Interpolation + Layernorm + GELU + Deformable Convolution v2          |           23.069908          |


### UNet Decoder scaling

Effect of Decoder scaling. We use a fixed number of 4 UNet Decoder blocks with output strides of \[16, 8, 4, 1]. We also use the ConvNeXT-Large encoder which already has a stronger capacity, larger number of parameters, but performance still scales very well with stronger Decoder in terms of parameters and number of output channels in high-resolution feature maps. This suggests we need a high resolution output feature map with a larger number of channels to achieve a competitive performance.

|                                                             |                              |
| :---------------------------------------------------------: | :--------------------------: |
| **Number of output channels in each****UNet Decoder block** | **Validation SNR on fold 0** |
|                       \[128,96,64,32]                       |           22.663811          |
|                       \[160,128,96,64]                      |           22.843121          |
|                      \[192,160,128,96]                      |           22.954165          |
|                      \[256,192,160,128]                     |           23.014694          |
|                      \[320,256,192,160]                     |           23.050228          |
|                      \[320,256,224,192]                     |           23.103617          |
|                      \[320,288,256,256]                     |           23.125187          |
|                      \[320,320,320,320]                     |         **23.163530**        |


### Input image & Output heatmap resolution

Effect of input/output resolution and UNet Decoder’s output strides. We use the same ConvNeXT-Small for all experiments, but change the first stem convolution’s stride (default to 4) to change the feature map’s resolution/stride accordingly.

|                      |                       |                                          |                                          |                              |
| :------------------: | :-------------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------: |
| **Input resolution** | **Output resolution** | **Encoder’s output feature map strides** | **Decoder’s output feature map strides** | **Validation SNR on fold 0** |
|        512x512       |        512x512        |              \[4, 8, 16, 32]             |              \[1, 4, 8, 16]              |           22.901585          |
|        512x512       |        512x512        |              \[2, 4, 8, 16]              |               \[1, 2, 4, 8]              |           23.108664          |
|        512x512       |        512x512        |                \[1,2,4,8]                |               \[1, 1, 2, 4]              |           23.191549          |
|        512x512       |       1024x1024       |              \[4, 8, 16, 32]             |      \[0.5, 1, 4, 8, 16] (5 blocks)      |           23.135838          |
|       1024x1024      |       1024x1024       |              \[4, 8, 16, 32]             |              \[1, 4, 8, 16]              |         **23.566664**        |


### Grid keypoints coordinate source 

Effect of grid keypoints coordinates. The Trained Keypoints Detection models actually produce much better keypoints coordinates in terms of both sub-pixel accuracy and consistency (e.g, constant shift), which is important for subsequent models to correct the alignment error.

|                                       |                               |
| :-----------------------------------: | :---------------------------: |
| **Grid keypoints coordinates source** | **fold0\@val/heatmap\_\_SNR** |
|              MINIMA-RoMa              |           22.550257           |
|      Trained Round 1 UNet models      |         **22.874523**         |
|      Trained Round 2 UNet models      |           22.866550           |

    
### Random Keypoints Offset Augmentation

Effect of Keypoints Offset Augmentation, where we sweep over the augmentation probabilities and offset sigma, where actual offset is sampled from a truncated normal distribution with mean=0 and the corresponding sigma (standard deviation). We start with sigma=0.4 pixels due to statistically analyzing our keypoints detection model’s predictions error on type 0001 images (the only type with known perfect keypoint coordinates). Interestingly, it’s also the best value we found so far.

|                              |                           |                              |
| :--------------------------: | :-----------------------: | :--------------------------: |
| **Augmentation probability** | **Offset sigma (pixels)** | **Validation SNR on fold 0** |
|         0.0 (disable)        |            N/A            |           22.929716          |
|             0.25             |            0.4            |           22.955214          |
|              0.5             |            0.4            |         **22.990671**        |
|              0.5             |            0.8            |           22.966988          |
|              0.5             |            1.2            |           22.937637          |
|             0.75             |            0.4            |           22.935484          |
|              1.0             |            0.4            |           22.912760          |


### Blank template as auxiliary input channel

Effect of using the blank template as auxiliary input channels, concatenate directly to the RGB lead cropping image. We use the stronger ConvNeXT-XLarge encoder in this experiment setup. Interestingly, using the grayscale template results in better performance than the RGB variant. The hypothesis using less auxiliary template channels acts as a regularization, encouraging model focus on actual RGB lead cropping channels, but also preserving almost all information of the template (grayscale is enough, RGB is somehow redundant). Also, grayscale templates could be better correlated with Black-White scanning image types.

|                                           |                              |
| :---------------------------------------: | :--------------------------: |
|  **Template as auxiliary input channels** | **Validation SNR on fold 0** |
|    Baseline with no template (disable)    |           23.132069          |
|    RGB Template (3 + 3 channels input)    |           23.156290          |
| Grayscale Template (3 + 1 channels input) |         **23.251383**        |



### Better rectification/warping method

Effect of better rectification method (4) using K=16 surrounding grid keypoints instead of the default method (2) using K=4. We found that using K=16 results in superior performance, especially when combined with Random Keypoint Offset augmentation.

|                                                                                                     |                              |
| :-------------------------------------------------------------------------------------------------: | :--------------------------: |
|                                       **Rectification method**                                      | **Validation SNR on fold 0** |
|                               Method (2) Piecewise Homography with K=4                              |           22.901585          |
|  Method (2) Piecewise Homography with K=4+ Random keypoints offset augmentation (p=0.5, sigma=0.4)  |           22.990671          |
|                              Method (4) Piecewise Homography with K=16                              |           23.086725          |
| Method (4) Piecewise Homography with K=16 + Random keypoints offset augmentation (p=0.5, sigma=0.4) |         **23.195719**        |
