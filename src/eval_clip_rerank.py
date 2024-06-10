import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

from utils import ParquetMetadataProvider, EmbeddingProvider, NaiveKNNIndex, load_clip
from metrics import MetricAverage, compute_retrieval_metrics

inat24_dir = 'data/inat24'
inquire_dir = 'data/inquire'
device = 'cpu'

rerank_df = pd.read_csv(os.path.join(inquire_dir, 'inquire_rerank.csv'))
queries = rerank_df['query'].unique()

# This dict maps from each model's short name (used to store info) to its full model name
models = {
    'vit-b-32': 'hf_clip:openai/clip-vit-base-patch32',
    'vit-l-14': 'hf_clip:openai/clip-vit-large-patch14',
    'vit-l-14-dfn': 'open_clip:ViT-L-14-quickgelu/dfn2b',
    'vit-h-14-378': 'open_clip:ViT-H-14-378-quickgelu/dfn5b',
    'siglip-so400m-14-384': 'open_clip:ViT-SO400M-14-SigLIP-384/webli',
    'siglip-vit-l-16-384': 'open_clip:ViT-L-16-SigLIP-384/webli',
}

print(f'{"Model":30s}\tmAP\tnDCG\tMRR')

for embs_name, clip_name in models.items():
    model, preprocess, tokenizer = load_clip(clip_name, use_jit=False, device=device)

    # Load embeddings. The pre-computed clip embeddings may not be stored in the same order.
    # To find the right embedding, we index by file name from the embedding metadata file
    embedding_provider = EmbeddingProvider(os.path.join('./data/inat24/embs', embs_name, "img_emb"))
    embedding_metadata = pd.Index(ParquetMetadataProvider(os.path.join(inat24_dir, "embs", embs_name, "metadata")).metadata_df['image_path'])
    
    metrics_avg = MetricAverage()
    for query in queries:
        query_df = rerank_df[rerank_df['query'] == query]
        
        text = tokenizer(query).to(device)
        with torch.no_grad(), torch.autocast(device):
            text_emb = model.encode_text(text).squeeze().float().cpu()
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
        
        # Get precomputed embeddings
        embeddings = []
        for _, image_row in query_df.iterrows():
            idx = embedding_metadata.get_loc(image_row['file_name'][:-4])
            emb = embedding_provider.get([idx]).squeeze()
            embeddings.append(emb)
            
        y_pred = (torch.from_numpy(np.array(embeddings, dtype=np.float32)) @ text_emb).numpy()
        y_true = query_df['relevant'].to_numpy()
            
        pr, rec, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
        metrics_avg.update([ap*100, ndcg*100, mrr])
        
    ap, ndcg, mrr = metrics_avg.avg
    print(f'{embs_name:30s}\t{ap:.1f}\t{ndcg:.1f}\t{mrr:.2f}')