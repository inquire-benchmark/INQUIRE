"""Run retrieval evaluation for the clip zero-shot baseline."""
import os
import numpy as np
import pandas as pd
import torch

from utils import TopKPredictionStore
from metrics import MetricAverage, compute_retrieval_metrics


device = 'cuda' if torch.cuda.is_available() else 'cpu'

inat24_dir = 'data/inat24'
inquire_dir = 'data/inquire'
cache_dir = 'data/top_k_cache'

# Load retrieval dataset
queries_df = pd.read_csv(os.path.join(inquire_dir, 'inquire_queries.csv'))
annotations_df = pd.read_csv(os.path.join(inquire_dir, 'inquire_annotations.csv'))
queries = queries_df['query_text'].values
print(f"Loaded dataset with {len(queries)} queries.")

# Pre-calculate top-k retrievals for all models
top_k_store = TopKPredictionStore(inat24_dir, cache_dir=cache_dir, device=device)
for model_name in list(top_k_store.models.keys()):
    _ = top_k_store.get_top_k(queries, model_name=model_name, k=1000)
        

print(f'k\t{"Model":30s}\tRecall\tPR\tmAP\tnormAP\tnDCG\tMRR')
for k in [25, 50, 100, 200, 500]:
    for model_name in list(top_k_store.models.keys()):
        top_ks = top_k_store.get_top_k(None, model_name, k=k)

        metrics_avg = MetricAverage()
        for query_row, top_k_data in zip(queries_df.iloc, top_ks):
            annotations = annotations_df[annotations_df['query_id'] == query_row['query_id']]
            pos_images = set(annotations['image_path'].values)

            y_pred = top_k_data["distances"].cpu().numpy()
            y_true = np.asarray([im['image_path']+".jpg" in pos_images for im in top_k_data["matches"]])
            
            pr, recall, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=len(pos_images))
            metrics_avg.update([pr, recall, ap, ndcg, mrr])
            
            # Use this line to print per-query metrics
            # print(f'{k:d}\t{query[:50]:50s}\t{recall:.4f} \t{pr:.4f}\t{ap:.4f}\t{ndcg:.4f}\t{mrr:.4f}')

        pr, recall, ap, ndcg, mrr = metrics_avg.avg
        #print(f'{k:d}\\t{model_name:30s}\\t{recall:.4f} \\t{pr:.4f}\\t{ap*100:.2f}\\t{nap*100:.2f}\\t{ndcg*100:.2f}\\t{mrr:.4f}')
        print(f'{k:d}\\t{model_name:30s}\\t{recall:.4f} \\t{pr:.4f}\\t{ap*100:.1f}\\t{ndcg*100:.1f}\\t{mrr:.2f}')
