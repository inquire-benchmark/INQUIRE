# Cached Results

We provide caches for the top-k results from CLIP models and inference from the multi-modal LLMs we use for reranking.

## CLIP top-k cache

For each CLIP model, we provide a cache containing the top 1000 images retrievals from all of iNat24 (scored via cosine similarity). Using these caches provide an easy place to start and makes it easy to replicate our results without needing to download all the iNat24 embedding and re-compute the top-k images. These caches can be downloaded [at this link](https://drive.google.com/drive/folders/1IM4EO6BCpEpHVbtnhN5Ef74GFNsk2l3e?usp=sharing).

## LLM ranking cache

INQUIRE evaluates using LLMs to rerank images, both for the INQUIRE-rerank task and as the second stage in the INQUIRE-fullrank, full-dataset retrieval task.

This style of reranking gets a score for each image, so the cache saves the response for each unique (image, prompt) pair.

This folder contains one such cached file for LLaVA-v1.5-13B, however a full list can be downloaded here (TODO)
