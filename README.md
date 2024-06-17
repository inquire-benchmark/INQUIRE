# INQUIRE

![INQUIRE teaser figure](assets/teaser.jpg)

[**🌐 Homepage**](https://inquire-benchmark.github.io/) | [**🖼️ Dataset**](https://github.com/inquire-benchmark/INQUIRE/tree/main/data/) 

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

## INQUIRE-Fullrank Evaluation

**INQUIRE-Fullrank** is the full-dataset retrieval task, starting from all 5 million images of iNat24. 

Evaluate full-dataset retrieval with different CLIP-style models:

```
python eval_clip_fullrank.py
```

## INQUIRE-Rerank Evaluation

**INQUIRE-Rerank** evaluates reranking performance by fixing an initial retrieval of 100 images for each query (from OpenClip's CLIP ViT-H-14-378). 

Evaluate reranking with different CLIP-style models:

```
python eval_clip_rerank.py
```
