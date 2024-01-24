# Image Derain
This repository records the proposed Image Derain methods based on Pytorch Library

> **Abstract:** *Many feature extraction components have been proposed for image deraining tasks, aiming to improve feature learning. However, few models have addressed the integration of multi-scale features from derain images. The fusion of multiple features at different scales in one model has the potential to significantly enhance the authenticity and detail of raingy images restoration. This study introduces a migratable multi-scale feature blending model, which is a progressive learning model based on detail dilation and texture extraction. First, the degraded image is sent to the detail dilation module, which is designed to increase the detailed outline and obtain the coarse image features. Second, the extracted feature maps are sent to the multi-scale feature extraction (MFE) module and the multiscale hybrid strategy (MHS) module for improved texture restoration. Third, the simple convolution modules are replaced by an optimized transformer model to more efficiently extract contextual features and multi-scale information in images. Finally, a progressive learning strategy is employed to incrementally restore the degraded images. Empirical results show that our proposed module for progressive restoration achieves near state-of-the-art performance in several rain removal tasks. In particular, our model exhibits better rain removal realism compared to state-of-the-art models.* 
<hr />

## Network Architecture

<img src = picture/Framework.jpg> 


## Trackers
Here is a list of the improved trackers.

* DIDMDN: Density-aware single image de-raining using a multi-stream dense network, CVPR, 2018. [[Paper]](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Density-Aware_Single_Image_CVPR_2018_paper.html)
* PReNet: Progressive image deraining networks: A better and simpler baseline, CVPR, 2019. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Ren_Progressive_Image_Deraining_Networks_A_Better_and_Simpler_Baseline_CVPR_2019_paper.html)
* MPRNet: Multi-Stage Progressive Image Restoration, CVPR, 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.html)  
* SPAIR: Spatially-Adaptive Image Restoration Using Distortion-Guided Networks, ICCV, 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Purohit_Spatially-Adaptive_Image_Restoration_Using_Distortion-Guided_Networks_ICCV_2021_paper.html)
* Restormer: Efficient transformer for high-resolution image restoration, 2022. [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html)  

## Findings
* Empirical results show that our proposed module for progressive inpainting achieves near state-of-the-art performance in several rain removal tasks.


## Description
This is the official repository of the paper 
[Progressive network based on detail scaling and texture extraction: A more general framework for image deraining]([https://arxiv.org/abs/2311.14631](https://www.sciencedirect.com/science/article/pii/S092523122301189X)https://www.sciencedirect.com/science/article/pii/S092523122301189X) 

# 🔧 Dependencies and Installation
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0](https://pytorch.org/)
```bash
pip install -r requirements.txt
```

## Training and Evaluation

Training and Testing instructions for Deraining, Motion Deblurring, Defocus Deblurring, and Denoising are provided in their respective directories. Here is a summary table containing hyperlinks for easy navigation:

<table>
  <tr>
    <th align="left">Task</th>
    <th align="center">Training Instructions</th>
    <th align="center">Testing Instructions</th>
    <th align="center">Restormer's Visual Results</th>
  </tr>
  <tr>
    <td align="left">Deraining</td>
    <td align="center"><a href="Deraining/README.md#training">Link</a></td>
    <td align="center"><a href="Deraining/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1HcLc6v03q_sP_lRPcl7_NJmlB9f48TWU?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Motion Deblurring</td>
    <td align="center"><a href="Motion_Deblurring/README.md#training">Link</a></td>
    <td align="center"><a href="Motion_Deblurring/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1qla3HEOuGapv1hqBwXEMi2USFPB2qmx_?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Defocus Deblurring</td>
    <td align="center"><a href="Defocus_Deblurring/README.md#training">Link</a></td>
    <td align="center"><a href="Defocus_Deblurring/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1V_pLc9CZFe4vN7c4SxtXsXKi2FnLUt98?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Gaussian Denoising</td>
    <td align="center"><a href="Denoising/README.md#training">Link</a></td>
    <td align="center"><a href="Denoising/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1rEAHUBkA9uCe9Q0AzI5zkYxePSgxYDEG?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Real Denoising</td>
    <td align="center"><a href="Denoising/README.md#training-1">Link</a></td>
    <td align="center"><a href="Denoising/README.md#evaluation-1">Link</a></td>
    <td align="center"><a href="https://drive.google.com/file/d/1CsEiN6R0hlmEoSTyy48nnhfF06P5aRR7/view?usp=sharing">Download</a></td>
  </tr>
</table>


## Updates 

`19/11/2023` Paper accepect at Neurocomputing! 🐣🐣🐣


# BibTeX
If you find DTPNet useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{huang2024progressive,
  title={Progressive network based on detail scaling and texture extraction: A more general framework for image deraining},
  author={Huang, Jiehui and Tang, Zhenchao and He, Xuedong and Zhou, Jun and Zhou, Defeng and Chen, Calvin Yu-Chian},
  journal={Neurocomputing},
  volume={568},
  pages={127066},
  year={2024},
  publisher={Elsevier}
}
