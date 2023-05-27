

# Distilling Style from Image Pairs for Global Forward and Inverse Tone Mapping


This repository provides code for Distilling Style from Image Pairs for Global Forward and Inverse Tone Mapping.

## Description

Many image enhancement or editing operations, such as forward and inverse tone mapping or color grading, do not have a unique solution, but instead a range of solutions, each representing a differentß style. Despite this, existing learning-based methods attempt to learn a unique mapping, disregarding this style. In this work, we show that information about the style can be distilled from collections of image pairs and encoded into a 2- or 3-dimensional vector. This gives us not only an efficient representation but also an interpretable latent space for editing the image style. We represent the global color mapping between a pair of images as a custom normalizing flow, conditioned on a polynomial basis of the pixel color. We show that such a network is more effective than PCA or VAE at encoding image style in low-dimensional space and lets us obtain an accuracy close to 40\,dB, which is about 7-10 dB improvement over the state-of-the-art methods.
For further information please refer to the [project webpage](https://www.cl.cam.ac.uk/research/rainbow/projects/distil_style/).

## Usage

The code runs in Python3 and Pytorch.

First install the dependencies:
* `Pytorch` 
* `Torchvision` 
* `Pillow`, please do `pip install pillow`  
* `Streamlit`, please do `pip install streamlit`
* `Bokeh`, please do `pip install bokeh`

You will also need to have the movie yuv files:

* `../../video/movie_name_4k/movie_name_960x540_420_2020_10b.yuv`
* `../../video/movie_name_hd/movie_name_960x540_420_709_8b.yuv`

# Test
* `python inference.py`

# Running Inference on Streamlit
* `streamlit run streamlit_3_Latents.py -- --frame_index <frame_number>`
* `streamlit run streamlit_2_Latents.py -- --frame_index <frame_number>`

Note: You need to use -- --frame_index to parse streamlit.

At every step on changing the attribute the inference runs on the GPU. Please check whether torch.cuda.is_available() is True.

The starting point of attributes in the slider are the average values across all frames. These need not be the best values for that particular frame.

This should automatically open a new browser tab with the UI.

## Citing

If using, please cite:

```
@inproceedings{mustafa2022distilling,
  title={Distilling Style from Image Pairs for Global Forward and Inverse Tone Mapping},
  author={Mustafa, Aamir and Hanji, Param and Mantiuk, Rafal},
  booktitle={Proceedings of the 19th ACM SIGGRAPH European Conference on Visual Media Production},
  pages={1--10},
  year={2022}
}
```
## Acknowledgement

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement N◦ 725253–EyeCode).

