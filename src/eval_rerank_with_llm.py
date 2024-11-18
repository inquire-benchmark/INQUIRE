"""This script evaluates using large language models for reranking on  INQUIRE-Rerank/
The data is automatically loaded from HuggingFace Hub, so you don't need to download 
anything yourself to run this evaluation."""

import argparse
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from datasets import load_dataset

from metrics import compute_retrieval_metrics
from lmm_utils import ModelWrapperWithCache, get_lmm_prompt, convert_gpt_response_to_preds

# Command line argument parser
parser = argparse.ArgumentParser(description='Run retrieval evaluation.')
parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                    help="Dataset split to evaluate on. Options: 'val', 'test'. Default is 'test'.")
args = parser.parse_args()

split = args.split
save_results_path = f'results_rerank_with_llm_{split}.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load INQUIRE-Rerank from HuggingFace
dataset = load_dataset("evendrow/INQUIRE-Rerank", split=('validation' if split == 'val' else 'test'))
queries = np.unique(dataset['query']).tolist()

model_names = [
    "Salesforce/blip2-flan-t5-xxl",
#    "Salesforce/instructblip-flan-t5-xxl", 
#    "google/paligemma-3b-mix-448",
#    "llava-hf/llava-1.5-13b-hf",
#    "llava-hf/llava-v1.6-mistral-7b-hf",
#    "llava-hf/llava-v1.6-34b-hf",
#    "Efficient-Large-Model/VILA1.5-13B",
#    "Efficient-Large-Model/VILA1.5-40b",
#    'openai-gpt4turbo20240409',
#    'openai-gpt4o20240806',
]

results = []
for model_name in model_names:
    model_cache = ModelWrapperWithCache(model_name, device)
    
    for query in tqdm(queries):
        query_ds = dataset.select(np.argwhere(np.asarray(dataset['query']) == query).squeeze())
        
        y_true = np.asarray(query_ds['relevant'])
        
        prompt = get_lmm_prompt(query, model_name)
        y_pred = model_cache.score_images(query_ds['inat24_file_name'], prompt, raw_images=query_ds['image'])
        if model_name.startswith('openai-gpt'):
            y_pred = convert_gpt_response_to_preds(y_pred)
        
        pr, recall, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
        results.append({
            'model': model_name,
            'query': query,
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


results_df = pd.DataFrame.from_dict(results)
pd.options.display.float_format = ' {:,.4f}'.format
print('='*40)
print(results_df.groupby('model').agg({'ap': 'mean', 'ndcg': 'mean', 'mrr': 'mean'}).sort_values('ap'))
print('='*40)

results_df.to_csv(save_results_path)
print("All done! Saved results to", save_results_path)
