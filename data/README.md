# INQUIRE and iNat2024

This directory stores the INQUIRE bnechmark, the iNaturalist 2024 dataset, and associated files such as embeddings and knn indices.

## Structure

To run the baselines, we expect the following structure:

```
data/
  |- inquire/       (<-- Already exists! No need to modify)
  |    |- ...
  |- inat24/        (<-- Only needed for the full-rank task)
  |    |- train/    
  |    |    | 00001_Animalia...
  |    |    | ...
  |    |- embs/
  |    |    |- vit-b-32/
  |    |    |- vit-l-14/
  |    |    |- ...
  |    |- train.json

```

## Download
All data is available for download at the following links:

INQUIRE:

- **[inquire_queries.csv (50MB)](#)**: The list of INQUIRE queries and their associated metadata.
- **[inquire_annotations.csv (50MB)](#)**: All (query, image) relevant pairs
- **[images_rerank.tar (10GB)](#)**: The subset of iNat24 used for the rerank challenge. This is an easier place to start!

iNat24:

- **[train.tar.gz (441GB)](https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.tar.gz)**: All iNat24 images, with 5,000,000 images and 10,000 classes. (`md5sum` hash: `2b1c7d7a023114c34cfccfb663c889a8`)
- **[train.json.tar.gz (438MB)](https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.json.tar.gz)**: The associated metadata for iNat24



Embeddings:
- **[vit-b-32.tar (XXGB)](#)**: Embeddings for the entire iNat24 dataset using the OpenCLIP ViT-B/32 model
- **[vit-l-14.tar (XXGB)](#)**: Embeddings for the entire iNat24 dataset using the OpenCLIP ViT-L/14 model
- **[vit-h-14-378.tar (XXGB)](#)**: Embeddings for the entire iNat24 dataset using the OpenCLIP ViT-H/14@378 model trained on DFB5B
