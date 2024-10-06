import json
import os
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import sklearn.metrics as metrics
import torch
from tqdm import tqdm

from utils import TopKPredictionStore
from metrics import compute_retrieval_metrics
from lmm_utils import ModelWrapperWithCache, get_lmm_prompt, convert_gpt_response_to_preds


# Command line argument parser
parser = argparse.ArgumentParser(description='Run retrieval evaluation.')
parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                    help="Dataset split to evaluate on. Options: 'val', 'test'. Default is 'test'.")
parser.add_argument('--k', type=int, default=50,
                    help="Top-k value for retrieval evaluation. Default is 50.")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

inat24_dir = 'data/inat24'
inquire_dir = 'data/inquire'
k = args.k
split = args.split
stage_one_method = 'vit-h-14-378'


# Load INQUIRE Fullrank
queries_df = pd.read_csv(os.path.join(inquire_dir, f'inquire_queries_{split}.csv'))
annotations_df = pd.read_csv(os.path.join(inquire_dir, 'inquire_annotations.csv'))
queries = queries_df['query_text'].values
print("# queries:", len(queries))


top_k_store = TopKPredictionStore(inat24_dir, cache_dir='./cache', device=device)

model_names = [
    "Salesforce/blip2-flan-t5-xxl",
#     "Salesforce/instructblip-flan-t5-xxl", 
#     "google/paligemma-3b-mix-448",
#     "llava-hf/llava-1.5-13b-hf",
#     "llava-hf/llava-v1.6-mistral-7b-hf",
#     "llava-hf/llava-v1.6-34b-hf",
#     "Efficient-Large-Model/VILA1.5-13B",
#     "Efficient-Large-Model/VILA1.5-40b",
#     'openai-gpt4turbo20240409',
#     'openai-gpt4o20240806',
]

results = []
for model_name in model_names:
    model_cache = ModelWrapperWithCache(model_name, device=device)

    top_ks = top_k_store.get_top_k(queries, stage_one_method, k=k)

    for query_row, top_k_data in zip(queries_df.iloc, top_ks):
        annotations = annotations_df[annotations_df['query_id'] == query_row['query_id']]
        pos_images = set(annotations['image_path'].values)
        
        y_true = np.asarray([im['image_path']+".jpg" in pos_images for im in top_k_data["matches"]])

        images = [im['image_path']+'.jpg' for im in top_k_data["matches"]]
        prompt = get_lmm_prompt(query_row.query_text, model_name)
        y_pred = model_cache.score_images(images, prompt, inat24_dir)
        if model_name.startswith('openai-gpt'):
            y_pred = convert_gpt_response_to_preds(y_pred)

        pr, recall, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=len(pos_images))
        results.append({
            'model': model_name,
            'query': query_row.query_text,
            'pr': pr * 100,
            'ap': ap * 100,
            'ndcg': ndcg * 100,
            'mrr': mrr
        })
        
        # If the model was loaded, inference was done, so save the cache
        if model_cache.model is not None:
            model_cache.save()
    
    # Clear the model from GPU to avoid OOM
    if model_cache.model is not None:
        del model_cache.model
        del model_cache.processor
        torch.cuda.empty_cache()

results_df = pd.DataFrame(results)
aggregated_results = results_df.drop(columns=['query']).groupby('model').mean()

print('='*40)
print(f'k = {k}')
print(aggregated_results)
print('='*40)

# Save results to CSV
output_csv = f'results_fullrank_two_stage_{split}_k{k}.csv'
print('Saving to:', output_csv)
results_df.to_csv(output_csv, index=False)