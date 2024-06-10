# INQUIRE and iNat2024

This directory stores the INQUIRE bnechmark, the iNaturalist 2024 dataset, and associated files such as embeddings and knn indices.

## Structure

To run the baselines, we expect the following structure:

```
data/
  |- inquire/ 
  |    |- inquire_queries.csv
  |    |- inquire_annotations.csv
  |    |- inquire_rerank.csv
  |- inat24/        
  |    |- train/   (<-- the iNat24 dataset)
  |    |    | 00001_Animalia...
  |    |    | ...
  |    |- train_rerank/  
  |    |    | 00001_Animalia...
  |    |- embs/
  |    |    |- vit-b-32/
  |    |    |- vit-l-14/
  |    |    |- ...
  |    |- train.json 
```

The files `train/` and `train.json` are the iNat24 dataset, which you only need if you want to do full-dataset retrieval.

If you're just interested in the reranking challenge, we provide `train_rerank`, a much smaller subset of iNat24 with just the top 100 retrievals for each task.

## Download
All data is available for download at the following links:

INQUIRE:

- **[inquire_queries.csv (50MB)](#)**: The list of INQUIRE queries and their associated metadata.
- **[inquire_annotations.csv (50MB)](#)**: All (query, image) relevant pairs

iNat24:

- **[train.tar.gz (441GB)](https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.tar.gz)**: All iNat24 images, with 5,000,000 images and 10,000 classes. (`md5sum` hash: `2b1c7d7a023114c34cfccfb663c889a8`)
- **[train_rerank.tar (10GB)](#)**: The subset of iNat24 used for the rerank challenge. This is an easier place to start!
- **[train.json.tar.gz (438MB)](https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.json.tar.gz)**: The associated metadata for iNat24


Embeddings:
- **[vit-b-32.tar (XXGB)](#)**: Embeddings for the entire iNat24 dataset using the OpenCLIP ViT-B/32 model
- **[vit-l-14.tar (XXGB)](#)**: Embeddings for the entire iNat24 dataset using the OpenCLIP ViT-L/14 model
- **[vit-h-14-378.tar (XXGB)](#)**: Embeddings for the entire iNat24 dataset using the OpenCLIP ViT-H/14@378 model trained on DFB5B
