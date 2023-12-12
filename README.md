# text-to-music

This repository displays a reproduction of the article [Moûsai: Text-to-Music Generation with Long-Context Latent Diffusion](https://arxiv.org/pdf/2301.11757v3.pdf) with some changes. The results were presented in **Topics in Data Science III – Hands on 
Machine Learning Project for Music Generation** classroom presented by the teacher Hélio Lopes.

## What is?

This is a training and evaluation adaptation of the [Moûsai](https://arxiv.org/pdf/2301.11757v3.pdf) algorithm, created by the authors Flavio Schneider, Ojasv Kamal, Zhijing Jin, Bernhard Schölkopf.

To learn more about how the model works, consider seeing the article and its notes in detail.

## Datasets

### MusicCaps

The MusicCaps dataset contains **5,521 music examples, each of which is labeled with an English aspect list and a free text caption written by musicians**.

See more in [MusicCaps (kaggle.com)](https://www.kaggle.com/datasets/googleai/musiccaps)

### Free Music Archive - FMA (Needs update)

An open and easily accessible dataset suitable for evaluating several tasks in MIR, a field concerned with browsing, searching, and organizing large music collections. The FMA aims to overcome this hurdle by providing 917 GiB and 343 days of Creative Commons-licensed audio from 106,574 tracks from 16,341 artists and 14,854 albums, arranged in a hierarchical taxonomy of 161 genres. It provides full-length and high-quality audio, pre-computed features, together with track- and user-level metadata, tags, and free-form text such as biographies. We here describe the dataset and how it was created, propose a train/validation/test split and three subsets, discuss some suitable MIR tasks, and evaluate some baselines for genre recognition. 

Code, data, and usage examples are available at https://github.com/mdeff/fma.

## Notebooks

* [MusicCaps Exploration](./notebooks/musiccaps-exploration.ipynb): A notebook that describes how the MusicCaps dataset was downloaded and used.

* [Diffusion Autoencoder](./notebooks/diffusion_autoencoder.v2.ipynb): A notebook that implements the training of a diffusion autoencoder.

* [Diffusion Model](./notebooks/diffusion_model.v1.ipynb): A notebook that implements the training and evaluation of a diffusion model.

## Usage

If you have a trained model, you can test the algorithm using the command below:

```bash
python prompt.py 'model.ckpt' "Your text prompt" --num_audios 1 --num_steps 100
```

|    Args     |               Description            |       Required       |
|:-----------:|:------------------------------------:|:--------------------:|
| model_file  | The name of the model file to use.   | Sim                  |
| prompt      | he text prompt for audio generation  | Sim                  |
| num_steps   | Number of steps for audio generation | Não, default: `100`  |
| num_audios  | Number of audios to generate         | Não, default: `2`    |

## References

* [Moûsai: Text-to-Music Generation with Long-Context Latent Diffusion](https://arxiv.org/pdf/2301.11757v3.pdf)

> Schneider, Flavio, Zhijing Jin, and Bernhard Schölkopf. "Mo\^ usai: Text-to-Music Generation with Long-Context Latent Diffusion." arXiv preprint arXiv:2301.11757 (2023).

* [MusicLM: Generating Music From Text](https://arxiv.org/abs/2301.11325)

> Agostinelli, Andrea, et al. "Musiclm: Generating music from text." arXiv preprint arXiv:2301.11325 (2023).

* [FMA: A dataset for music analysis.](https://arxiv.org/abs/1612.01840)

> Defferrard, Michaël, et al. "FMA: A dataset for music analysis." arXiv preprint arXiv:1612.01840 (2016).

* [Exploring the limits of transfer learning with a unified text-to-text transformer (t5-base)](https://dl.acm.org/doi/abs/10.5555/3455716.3455856)

> Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." The Journal of Machine Learning Research 21.1 (2020): 5485-5551.

## Issues

Feel free to submit issues and enhancement requests.

## Contribution

1. Fork the project
2. Create a _branch_ for your modification (`git checkout -b my-new-resource`)
3. Do the _commit_ (`git commit -am 'Adding a new resource...'`)
4. _Push_ (`git push origin my-new-resource`)
5. Create a new _Pull Request_ 