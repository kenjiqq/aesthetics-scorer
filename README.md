# aesthetics-scorer

Predicts aesthetic scores for images. Trained on Stable Hord user ratings of Stable Diffusion generated images.

## Visualized results

### Different clip models on validation split of dataset
* [openclip_vit_bigg_14](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/visualize-aesthetics_scorer_openclip_vit_bigg_14.html)
* [openclip_vit_h_14](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/visualize-aesthetics_scorer_openclip_vit_h_14.html)
* [openclip_vit_l_14](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/visualize-aesthetics_scorer_openclip_vit_l_14.html)

### Openclip_vit_h_14 model on subset of laion5b
* [laion5b](https://htmlpreview.github.io/?https://github.com/kenjiqq/aesthetics-scorer/blob/main/visualize/laion5b-visualize.html)

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
