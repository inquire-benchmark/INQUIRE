# INQUIRE

![INQUIRE teaser figure](assets/teaser.jpg)

[**🌐 Homepage**](https://inquire-benchmark.github.io/) | [**🖼️ Dataset**](https://github.com/inquire-benchmark/INQUIRE/tree/main/data/) | [**🤗 HuggingFace**](https://huggingface.co/datasets/evendrow/INQUIRE-Rerank)

INQUIRE is a benchmark for expert-level natural world image retrieval queries.

**Please note that this repository is preliminary. Both the code and dataset will be updated.**


## 🔔 News
- **🚀 [2024-06-07]** INQUIRE is up! 

## Download

The **INQUIRE benchmark** and **the iNaturalist 2024 dataset (iNat24)** are available for public download. Please find information and download links [here](data/README.md).

## Setup

If you'd like, you can create a new environment in which to set up the repo:
```bash
conda create -n inquire python=3.10
conda activate inquire
```

Then, install the dependencies:
```bash
pip install -r requirements.txt
```

Our evaluations use pre-computed CLIP embeddings over iNat24. If you'd like to replicate our evaluations or just work with these embeddings, please download them [here](data/README.md). 

## INQUIRE Fullrank Evaluation

**INQUIRE-Fullrank** is the full-dataset retrieval task, starting from all 5 million images of iNat24. We evaluate **one-stage** retrieval, using similarity search with CLIP-style models, and **two-stage** retrieval, where after the initial retrieval, a large multi-modal model is used to rerank the images.

### One-stage retrieval with CLIP-style models

To evaluate full-dataset retrieval with different CLIP-style models, you don't necessarily need all 5 million images, but rather their embeddings. You can download our pre-computed embeddings for a variety of models from [here](data/README.md). Then, use the following command to evaluate CLIP retrieval:

```
python src/eval_fullrank.py --split test --k 50
```

### Two-stage retrieval

After the first stage, we can use large multi-modal models to re-rank the top k retrievals to improve results. This stage requires access to the iNat24 images, which you can download [here](data/README.md). To run the second stage retrieval, use the following command:

```
python src/eval_fullrank_two_stage.py --split test --k 50
```


## INQUIRE-Rerank Evaluation

**We recommend starting with INQUIRE-Rerank, as it is much smaller. INQUIRE-Rerank is available on [🤗 HuggingFace](https://huggingface.co/datasets/evendrow/INQUIRE-Rerank)!**

**INQUIRE-Rerank** evaluates reranking performance by fixing an initial retrieval of 100 images for each query (from OpenClip's CLIP ViT-H-14-378). This also makes it easier to experiment with, as you only need a significantly smaller set of images.

Evaluate reranking with different CLIP-style models:

```
python eval_clip_rerank_hf.py
```

Evaluate reranking with large multi-modal models, such as LLaVA-34B:

TODO

Since inference can take a long time, we've pre-computed the outputs for all large multi-modal models we work with! You can download these [here](data/README.md).
