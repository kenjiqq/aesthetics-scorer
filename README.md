# aesthetics-scorer

Predicts aesthetic scores for images. Trained on AI Horde community ratings of Stable Diffusion generated images.

[Huggingface demo](https://huggingface.co/spaces/kenjiqq/aesthetics-scorer)

## Visualized results

### Validation split of diffusiondb dataset

#### OpenClip models

* [ratings](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/diffusiondb/visualize-aesthetics_scorer_rating_openclip_vit_bigg_14.html) 
| [artifacts](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/diffusiondb/visualize-aesthetics_scorer_artifacts_openclip_vit_bigg_14.html) 
| openclip_vit_bigg_14
* [ratings](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/diffusiondb/visualize-aesthetics_scorer_rating_openclip_vit_h_14.html) 
| [artifacts](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/diffusiondb/visualize-aesthetics_scorer_artifacts_openclip_vit_h_14.html) 
| openclip_vit_h_14
* [ratings](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/diffusiondb/visualize-aesthetics_scorer_rating_openclip_vit_l_14.html) 
| [artifacts](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/diffusiondb/visualize-aesthetics_scorer_artifacts_openclip_vit_l_14.html) 
| openclip_vit_l_14

#### Convnext models
* [ratings](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/diffusiondb/visualize-aesthetics_rating_convnext_large_2e_b2e.html)
| [artifacts](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/diffusiondb/visualize-aesthetics_artifacts_convnext_large_2e_b2e.html) 
| convnext_large
* [ratings](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/diffusiondb/visualize-aesthetics_rating_realfake_2e_b2e.html)
| [artifacts](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/diffusiondb/visualize-aesthetics_artifacts_realfake_2e_b2e.html) 
| realfake

### Subset of laion5b

####  OpenClip models

* [ratings](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/laion/visualize-laion5b-rating-openclip_vit_h_14.html) 
| [artifacts](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/laion/visualize-laion5b-artifacts-openclip_vit_h_14.html)
| openclip_vit_h_14

#### Convnext models

* [ratings](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/laion/visualize-laion5b-rating-aesthetics_rating_convnext_large_2e_b2e.html) 
| [artifacts](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/laion/visualize-laion5b-artifacts-aesthetics_artifacts_convnext_large_2e_b2e.html)
| convnext_large
* [ratings](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/laion/visualize-laion5b-rating-aesthetics_rating_realfake_2e_b2e.html) 
| [artifacts](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/laion/visualize-laion5b-artifacts-aesthetics_artifacts_realfake_2e_b2e.html)
| realfake


## Usage 


Model files in aesthetics_scorer/models folder

Simple gradio demo
```bash
python aesthetics_scorer/demo.py
```

## Train

### Prepare dataset

1. dataset-process/dataset_downloader.py downloads zipped diffusiondb dataset images, change path to where you want it stored (~200gb)
2. dataset-process/dataset_parquet_files.py downloads dataset parquet files and sets up train and validation splits
3. dataset-process/dataset_image_extract.py extract the rated images from the zipped dataset files
4. dataset-process/clip_encode_dataset.py precomputes clip embeddings for all rated images (change config if you don't need the different clip versions)

If 1) is already downloaded then 2) can be rerun to update dataset parquet files and 3) and 4) will only perform work on new images needed that hasn't already been processed.

### Training
In aesthetics_scorer/train.py change whatever configs you want. Mostly importantly EMBEDDING_FILE to whatever embeddingfile you preprocessed.
There are a bunch of different hyperparams that can be changed.

```bash
python aesthetics_scorer/train.py
```


## Credits
* Inspired by https://github.com/christophschuhmann/improved-aesthetic-predictor
* Image dataset https://poloclub.github.io/diffusiondb/
* Image ratings by https://aihorde.net/

```
@article{wangDiffusionDBLargescalePrompt2022,
  title = {Large-Scale Prompt Gallery Dataset for Text-to-Image Generative Models},
  author = {Wang, Zijie J. and Montoya, Evan and Munechika, David and Yang, Haoyang and Hoover, Benjamin and Chau, Duen Horng},
  year = {2022},
  journal = {arXiv:2210.14896 [cs]},
  url = {https://arxiv.org/abs/2210.14896}
}
```
```
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```
