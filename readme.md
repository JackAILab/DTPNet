# Progressive Network based on Detail Scaling and Texture Extraction: A More General Framework for Image Derain 

Jiehui Huang, Haofan Huang, Zhenchao Tang, Lishan Lin, Defeng Zhou, Weicheng Lv, Yucheng Li, Kang Yuan†, Zhong Pei, Calvin Yu-Chian Chen

<hr />

**Abstract**: Many effective components have been proposed for feature learning for image-acquired rain removal tasks. However, few models consider how to combine multi-scale features of these rainy images. The effective coupling between multi-scale features in the image derain model can further improve the authenticity and detail of image derain. In this paper, we propose a migratable multi-scale feature mixture model, which is a progressive learning model based on detail dilation and texture extraction. First, the degraded image will be sent to the detail expansion module designed by us to enlarge the detail outline and obtain the rough features of the image. Then, the extracted feature maps will be sent to the multi-scale feature extraction (MFE) module and multi-scale hybrid strategy (MHS)module for refined texture restoration. We then replace simple convolution modules with an optimized Transformer model to better extract contextual features and multi-scale information in images.** **Finally, we use a progressive learning strategy to inpaint the degraded images step by step. Experimental results demonstrate that our proposed progressive inpainting module can almost achieve state-of-the-art performance in a variety of rain removal tasks. Specially, our model also has a better advantage in terms of image derain realism than state-of-the-art models.

<hr />

# Network Architecture

<img src = "https://github.com/ffdffd/MixDTPNet/blob/GITHUB/picture/Network.jpg?raw=true">

## Installation requirement

```
pip install -r requirements.txt
```



# Results

## Visualization result on 200L

<img src = "https://github.com/ffdffd/MixDTPNet/blob/GITHUB/picture/Result.jpg?raw=true">

## Visualization result on Rain200H, Rai200L Dataset

<img src = "https://github.com/ffdffd/MixDTPNet/blob/GITHUB/picture/result2.jpg?raw=true">

## Visualization result on RainDrop, SPA, RID Dataset

<img src = "https://github.com/ffdffd/MixDTPNet/blob/GITHUB/picture/results.jpg?raw=true">



## Result on 100L, 100H

<img src = "https://github.com/ffdffd/MixDTPNet/blob/GITHUB/picture/Result_100.jpg?raw=true">



## Result on 200L, 200H

<img src = "https://github.com/ffdffd/MixDTPNet/blob/GITHUB/picture/result_200.jpg?raw=true">

### Result on SPA

<img src = "https://github.com/ffdffd/MixDTPNet/blob/GITHUB/picture/result_SPA.jpg?raw=true">



