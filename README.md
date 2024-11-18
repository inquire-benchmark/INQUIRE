# INQUIRE

![INQUIRE teaser figure](assets/teaser.jpg)

[**🌐 Homepage**](https://inquire-benchmark.github.io/) | [**🖼️ Dataset**](https://github.com/inquire-benchmark/INQUIRE/tree/main/data/) | [**🤗 HuggingFace**](https://huggingface.co/datasets/evendrow/INQUIRE-Rerank) | [**📖 Paper**](https://arxiv.org/abs/2411.02537)

INQUIRE is an expert-level text-to-image retrieval benchmark designed to challenge multi-modal models. 


**Please note that this repository is preliminary. Both the code and dataset will be updated soon.**


## 🔔 News
- **🚀 [2024-11-06]** The paper for INQUIRE is up on arXiv! Check it out [here](https://arxiv.org/abs/2411.02537).
- **🚀 [2024-10-08]** INQUIRE was accepted to NeurIPS 2024 (Datasets and Benchmarks Track)!
- **🚀 [2024-06-07]** INQUIRE is up!


## 🌟 Key Features
- Large (5 million images) and exhaustively annotated (1-1.5k relevant images per query)
- Queries come from experts (e.g., ecologists, biologists, ornithologists, entomologists, oceanographers)
- Supports **two-stage retrieval** with CLIP models and reranking with large multimodal models.
- Includes pre-computed embeddings and model outputs for faster evaluation.



## Download

The **INQUIRE benchmark** and **the iNaturalist 2024 dataset (iNat24)** are available for public download. Please find information and download links [here](data/README.md).

## Setup

Clone the repository and navigate into it:
```bash
git clone https://github.com/inquire-benchmark/INQUIRE.git
cd INQUIRE
```

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

**We recommend starting with INQUIRE-Rerank, as it is much smaller and easier to work with. INQUIRE-Rerank is available on [🤗 HuggingFace](https://huggingface.co/datasets/evendrow/INQUIRE-Rerank)!**

**INQUIRE-Rerank** evaluates reranking performance by fixing an initial retrieval of 100 images for each query (from OpenClip's CLIP ViT-H-14-378). For each query (e.g. _A mongoose standing upright alert_), your task is to re-order the 100 images so that more of the relevant images are at the "top" of the reranked order. 

### Requirements

There are no extra requirements for evaluating INQUIRE-Rerank! The data will automatically download from HuggingFace if you don't already have it. 

### Reranking with large multi-modal models

Evaluate reranking performance with large multi-modal models such as LLaVA-34B:

```
python src/eval_rerank_with_llm.py --split test
```

Since inference can take a long time, we've pre-computed the outputs for all large multi-modal models we work with! You can download these [here](cache/README.md).

## Citation
If you use INQUIRE or find our work helpful, please consider starring our repo and citing our paper. Thanks!

```
@article{vendrow2024inquire,
  title={INQUIRE: A Natural World Text-to-Image Retrieval Benchmark}, 
  author={Vendrow, Edward and Pantazis, Omiros and Shepard, Alexander and Brostow, Gabriel and Jones, Kate E and Mac Aodha, Oisin and Beery, Sara and Van Horn, Grant},
  journal={NeurIPS},
  year={2024},
}
```
