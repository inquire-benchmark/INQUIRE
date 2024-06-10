"""This script runs evals for CLIP models on INQUIRE-Rerank, the reranking task.
The data is automatically loaded from HuggingFace Hub, so you don't need to download 
anything yourself to run this evaluation."""
import os
import json
import shutil
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from collections import defaultdict 

from utils import load_clip
from metrics import MetricAverage, compute_retrieval_metrics


device = 'cuda'
batch_size = 256
num_workers = 8
save_results_path = 'results_clip_rerank.csv'

# Load INQUIRE-Rerank from HuggingFace
dataset = load_dataset("evendrow/INQUIRE-Rerank", split="train")
queries = np.unique(dataset['query']).tolist()

all_models = {
    'vit-b-32': 'hf_clip:openai/clip-vit-base-patch32',
    'wildclip-t1': 'wildclip_vitb16_t1',
    'wildclip-t1t7-lwf': 'wildclip_vitb16_t1t7_lwf',
    'bioclip': 'bioclip',
    'rn50': 'open_clip:RN50/openai',
    'rn50x16': 'open_clip:RN50x16/openai',
    'vit-b-16': 'hf_clip:openai/clip-vit-base-patch16',
    'vit-l-14': 'hf_clip:openai/clip-vit-large-patch14',
    'vit-b-16-dfn': 'open_clip:ViT-B-16/dfn2b',
    'vit-l-14-dfn': 'open_clip:ViT-L-14-quickgelu/dfn2b',
    'vit-h-14-378': 'open_clip:ViT-H-14-378-quickgelu/dfn5b',
    'siglip-vit-l-16-384': 'open_clip:ViT-L-16-SigLIP-384/webli',
    'siglip-so400m-14-384': 'open_clip:ViT-SO400M-14-SigLIP-384/webli',
}

results = []
for title, clip_name in all_models.items():
    model, preprocess, tokenizer = load_clip(clip_name, use_jit=False, device=device)

    # Efficiently compute image embeddings in batches
    def collate_transform(examples):
        pixel_values = torch.cat([preprocess(ex["image"]).unsqueeze(0) for ex in examples])
        ids = [ex['inat24_image_id'] for ex in examples]
        return pixel_values, ids
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_transform, num_workers=num_workers)

    image_emb_cache = {}
    for images, ids in tqdm(dataloader, total=len(dataset)//batch_size):
        with torch.no_grad(), torch.autocast(device):
            image_embs = model.encode_image(images.to(device)).cpu()
            image_embs /= image_embs.norm(dim=-1, keepdim=True)
        image_emb_cache.update(dict(zip(ids, image_embs)))


    # Score images for each query by embedding similarity
    metrics_avg = MetricAverage()
    for query in queries:
        query_ds = dataset.select(np.argwhere(np.asarray(dataset['query']) == query).squeeze())

        text = tokenizer(query).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_emb = model.encode_text(text).squeeze().cpu()
            text_emb /= text_emb.norm(dim=-1, keepdim=True)

        image_embs = torch.stack([image_emb_cache[image_id] for image_id in query_ds['inat24_image_id']])
        y_pred = (image_embs.float() @ text_emb.float()).numpy()
        y_true = np.asarray(query_ds['relevant'])

        pr, rec, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
        metrics_avg.update([ap*100, ndcg*100, mrr])
        results.append(dict(model=title, query=query, ap=ap*100, ndcg=ndcg*100, mrr=mrr))

    ap, ndcg, mrr = metrics_avg.avg
    print(f'{title:30s}\t{ap:.1f}\t{ndcg:.1f}\t{mrr:.2f}')

results_df = pd.DataFrame.from_dict(results)
pd.options.display.float_format = ' {:,.2f}'.format
print(results_df.groupby('model').agg({'ap': 'mean', 'ndcg': 'mean', 'mrr': 'mean'}).sort_values('ap'))

results_df.to_csv(save_results_path)
print("All done! Saved results to", save_results_path)