"""Run retrieval evaluation for the clip zero-shot baseline."""
import os
import numpy as np
import pandas as pd
import torch
import argparse

from utils import TopKPredictionStore
from metrics import MetricAverage, compute_retrieval_metrics

# Command line argument parser
parser = argparse.ArgumentParser(description='Run retrieval evaluation.')
parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                    help="Dataset split to evaluate on. Options: 'val', 'test'. Default is 'test'.")
parser.add_argument('--k', type=int, default=50,
                    help="Top-k value for retrieval evaluation. Default is 50.")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

inat24_dir = 'data/inat24'
inquire_dir = 'data/inquire'
cache_dir = 'cache'
k = args.k
split = args.split

# Load INQUIRE Fullrank
queries_df = pd.read_csv(os.path.join(inquire_dir, f'inquire_queries_{split}.csv'))
annotations_df = pd.read_csv(os.path.join(inquire_dir, 'inquire_annotations.csv'))
queries = queries_df['query_text'].values
print(f"Loaded dataset with {len(queries)} queries.")

# Pre-calculate top-k retrievals for all models
top_k_store = TopKPredictionStore(inat24_dir, cache_dir=cache_dir, device=device)
for model_name in list(top_k_store.models.keys()):
    _ = top_k_store.get_top_k(queries, model_name=model_name, k=1000)
        
results = []
for model_name in list(top_k_store.models.keys()):
    top_ks = top_k_store.get_top_k(queries, model_name, k=k)

    for query_row, top_k_data in zip(queries_df.iloc, top_ks):
        annotations = annotations_df[annotations_df['query_id'] == query_row['query_id']]
        pos_images = set(annotations['image_path'].values)

        y_pred = top_k_data['distances'].cpu().numpy()
        y_true = np.asarray([im['image_path']+".jpg" in pos_images for im in top_k_data["matches"]])

        pr, recall, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=len(pos_images))
        results.append({
            'model': model_name,
            'query': query_row.query_text,
            'pr': pr * 100,
            'ap': ap * 100,
            'ndcg': ndcg * 100,
            'mrr': mrr
        })

# Create results DataFrame and print aggregated metrics
results_df = pd.DataFrame.from_dict(results)
aggregated_results = results_df.drop(columns=['query']).groupby('model').mean()

print('='*40)
print(f'k = {k}')
print(aggregated_results)
print('='*40)

# Save results to CSV
output_csv = f'results_fullrank_{split}_k{k}.csv'
print('Saving to:', output_csv)
results_df.to_csv(output_csv, index=False)