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

### INQUIRE

- **[inquire_queries_val.csv (50MB)](https://github.com/inquire-benchmark/INQUIRE/tree/main/data/inquire)**: The list of 50 INQUIRE queries and their associated metadata, in the validation split.
- **[inquire_queries_test.csv (50MB)]([#](https://github.com/inquire-benchmark/INQUIRE/tree/main/data/inquire)**: The list of 200 INQUIRE queries and their associated metadata, in the test split.
- **[inquire_annotations.csv (50MB)](https://github.com/inquire-benchmark/INQUIRE/tree/main/data/inquire)**: All (query, image) relevant pairs

### iNat24

- **[train.tar.gz (441GB)](https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.tar.gz)**: All iNat24 images, with 5,000,000 images and 10,000 classes. (`md5sum` hash: `2b1c7d7a023114c34cfccfb663c889a8`)
- **[train.json.tar.gz (438MB)](https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.json.tar.gz)**: The associated metadata for iNat24


### Embeddings

We pre-computed CLIP embeddings for all iNat24 images, which are used for INQUIRE Fullrank retrieval. The embeddings for all the models we test are available [at this link](https://drive.google.com/drive/folders/1remNGZdc08B7i-Xg3oAaY68fnJ3QXyWm?usp=drive_link), and we list a few here as well:

- **[vit-b-16.tar (3.8GB)](https://drive.google.com/file/d/1JW-Z24zbcBuCb5bGRxMECUoIxH8Kmg__/view?usp=drive_link)**: Embeddings for the entire iNat24 dataset using the OpenAI's ViT-B/16 model
- **[vit-l-14.tar (5.6GB)](https://drive.google.com/file/d/1j5chxOkYq8WWsnFL8parpJwPXio-RkRX/view?usp=drive_link)**: Embeddings for the entire iNat24 dataset using OpenAI's ViT-L/14 model
- **[vit-h-14-378.tar (7.4GB)](https://drive.google.com/file/d/1QABd-7VpjzaOP7v1Kbf6Rgmgd5ujkArR/view?usp=drive_link)**: Embeddings for the entire iNat24 dataset using the OpenCLIP ViT-H/14@378 model trained on DFB5B
- **[siglip-so400m-14-384.tar (8.3GB)](https://drive.google.com/file/d/1VMBuA1KbSItKOV2kh1ag6RnUrQvdkEvS/view?usp=drive_link)**: Embeddings for the entire iNat24 dataset using Google's SigLIP-SO400m-14@384 model trained on WebLi.

## INQUIRE Data Format

Queries and annotations for INQUIRE are stored in two files.

**inquire_queries_val.csv** and **inquire_queries_test.csv** list queries and their associated metadata, with the following columns:
- `query_id`: The unique ID of the query
- `query_text`: The query string (e.g., _A mongoose standing upright alert_)
- `supercategory`: The supercategory of the query, one of Appearance Behavior, Context, or Species
- `category`: The fine-grained categorization of the query (e.g., _Tracking and Identification_)
- `iconic group`: A broad taxonomic grouping of the query (e.g., _Mammals_, _Birds_, _Fungi_) matching iNat21 and the iNaturalist platform.

**inquire_annotations.csv** lists all the annotations for both the val and test split. Each annotation is a (query, image) pair where the image is a relavant match for the query.
- `query_id`: The ID of the query, corresponding to a single row in either **inquire_queries_val.csv** or **inquire_queries_test.csv**
- `image_id`: The ID of the image, which corresponds to and entry in the iNat24 metadata file
- `image_path`: The path to the image within the iNat24 dataset (e.g., _train/04686_Animalia_Chordata_Mammalia_Carnivora_Herpestidae_Mungos_mungo/903be103-954f-409b-81f1-82d4478928f0.jpg_)
