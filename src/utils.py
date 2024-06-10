import os

import json
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import pickle
import torch

# https://github.com/data2ml/all-clip
# pip install all-clip==1.2.0
from all_clip import load_clip as load_vanilla_clip


def load_clip(clip_name, device, **args):
    """Wrapper around all_clip load_clip which adds support for wildcilp and bioclip"""
    
    if clip_name.startswith("wildclip"):
        import clip
        from clip import tokenize
        wildclip_path = f"./scripts/wildclip/models/{clip_name}.pth"
        model_dict = torch.load(wildclip_path)
        clip_model, preprocess = clip.load('ViT-B/16', device=device)
        state_dict_fixed = { k.split('model.clip_model.')[-1]: v for k, v in model_dict['state_dict'].items()}
        clip_model.load_state_dict(state_dict_fixed, strict=False)
        return clip_model, preprocess, tokenize        
    elif clip_name == "bioclip":
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        clip_model.to(device)
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
        return clip_model, preprocess, tokenizer        
    else:
        return load_vanilla_clip(clip_name, device=device, **args)


# From clip-retrieval
# https://github.com/rom1504/clip-retrieval/blob/main/clip_retrieval/clip_back.py#L521
class ParquetMetadataProvider:
    """The parquet metadata provider provides metadata from contiguous ids using parquet"""

    def __init__(self, parquet_folder):
        data_dir = Path(parquet_folder)
        self.metadata_df = pd.concat(
            pd.read_parquet(parquet_file) for parquet_file in sorted(data_dir.glob("*.parquet"))
        )

    def get(self, ids, cols=None):
        if cols is None:
            cols = self.metadata_df.columns.tolist()
        else:
            cols = list(set(self.metadata_df.columns.tolist()) & set(cols))

        return [self.metadata_df[i : (i + 1)][cols].to_dict(orient="records")[0] for i in ids]

    def __len__(self):
        return len(self.metadata_df)

    
class EmbeddingProvider:
    """Serves memory-mapped embeddings from an generated embedding directory."""
    def __init__(self, embedding_folder):
        data_dir = Path(embedding_folder)
        emb_files = sorted(data_dir.glob("*.npy"))
        self.embs = [np.load(f, mmap_mode='r') for f in emb_files]
        self.size = sum([len(emb) for emb in self.embs])

        print("Loaded memory-mapped embeddings with total:", self.size)

    def get_one(self, idx):
        if idx < 0:
            raise ValueError("Index must be nonnegative")
        for emb in self.embs:
            if idx >= len(emb):
                idx -= len(emb)
            else:
                return emb[idx]
        raise ValueError(f"index not found: {idx}")

    def get(self, idxs):
        results = []
        for idx in idxs:
            results.append(self.get_one(idx))
        return np.asarray(results)
    

class NaiveKNNIndex:
    def __init__(self, data_path, model_name, device='cpu'):
        emb_files_paths = sorted(glob.glob(os.path.join(data_path, 'embs', f'{model_name}/img_emb/img_emb_*.npy')))

        print('Initializing index with', len(emb_files_paths), '.npy files')
        self.all_embs = []
        for emb_file in emb_files_paths:
            embs = torch.from_numpy(np.load(emb_file, allow_pickle=True)).to(device)
            self.all_embs.append(embs)
        print('... Done')

    def search(self, query, k):
        query = query.squeeze()
        assert len(query.shape) == 1, "Embedding should be 1-dimensional"

        all_scores = []
        for embs in self.all_embs:
            all_scores.append(embs.float() @ query.float())
        
        scores = torch.cat(all_scores)
        assert len(scores.shape) == 1, "???"

        indices = torch.flip(scores.argsort(), dims=(0,))[:k]
        distances = scores[indices]
        return distances, indices


def load_metdata(index_folder):
    parquet_folder = index_folder + "/metadata" 
    metadata_provider = ParquetMetadataProvider(parquet_folder)
    return metadata_provider


class TopKPredictionStore:
    """Cache top K predictions for variety of models and Ks"""

    def __init__(self, data_dir, cache_dir='data/top_k_cache', device='cpu'):
        self.data_dir = data_dir
        self.device = device
        self.cache_dir = cache_dir
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

        self.models = {
            'vit-b-32': {
                'clip_name': 'hf_clip:openai/clip-vit-base-patch32'
            },
            'wildclip-t1': {
                'clip_name': 'wildclip_vitb16_t1'
            },
            'wildclip-t1t7-lwf': {
                'clip_name': 'wildclip_vitb16_t1t7_lwf'
            },
            'bioclip': {
                'clip_name': 'bioclip'
            },
            'rn50': {
                'clip_name': 'open_clip:RN50/openai'
            },
            'rn50x16': {
                'clip_name': 'open_clip:RN50x16/openai'
            },
            'vit-b-16': {
                'clip_name': 'hf_clip:openai/clip-vit-base-patch16'
            },
            'vit-l-14': {
                'clip_name': 'hf_clip:openai/clip-vit-large-patch14'
            },
            'vit-b-16-dfn': {
                'clip_name': 'open_clip:ViT-B-16/dfn2b',
            },
            'vit-l-14-dfn': {
                'clip_name': 'open_clip:ViT-L-14-quickgelu/dfn2b',
            },
            'vit-h-14-378': {
                'clip_name': 'open_clip:ViT-H-14-378-quickgelu/dfn5b'
            },
            'siglip-vit-l-16-384': {
                'clip_name': 'open_clip:ViT-L-16-SigLIP-384/webli'
            },
            'siglip-so400m-14-384': {
                'clip_name': 'open_clip:ViT-SO400M-14-SigLIP-384/webli'
            },
        }
        self.current_model = None

    def load_model(self, model_name):
        if self.current_model != model_name:
            index_path = os.path.join(self.data_dir, 'embs', model_name)
            clip_name = self.models[model_name]['clip_name']

            print('Loading model:', model_name)
            print('  >> With index at path:', index_path)
            print('  >> With CLIP model name:', clip_name)

            # Load index, metadata, and model
            self.index = NaiveKNNIndex(self.data_dir, model_name, self.device)
            self.metadata = load_metdata(index_path)
            model, preprocess, tokenizer = load_clip(clip_name, use_jit=False, device=self.device) #, device="cuda:0"

            self.current_model = model_name
            self.model = model
            self.tokenizer = tokenizer

    def get_top_k(self, queries, model_name, k, recalculate=False, from_k=1000):
        if model_name not in self.models.keys():
            raise ValueError(f'Model with name {model_name} not found.')
        if from_k < k:
            raise ValueError(f"Trying to get top {k} from a file that only contains top {from_k}!")

        model_name_sanitized = model_name.replace(":", "__").replace("/", "--")
        cache_path = os.path.join(self.cache_dir, f"{model_name_sanitized}--top-{from_k}.pkl")
        
        if os.path.exists(cache_path) and not recalculate:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            # return data
        
            data_subset = []
            for top_k_query in data:
                data_subset.append({
                    "query": top_k_query["query"],
                    "distances": top_k_query["distances"][:k],
                    "indices": top_k_query["indices"][:k],
                    "matches": top_k_query["matches"][:k]
                })
            return data_subset

        else:
            print('No cache file found, or forced to recalculate. Computing top-k...')

            self.load_model(model_name)

            data = []
            for query in queries:
                print("    > "+query)
                # Get text embedding
                text = self.tokenizer(query).to(self.device)
                with torch.no_grad(): #, torch.cuda.amp.autocast():
                    text_emb = self.model.encode_text(text)
                    text_emb /= text_emb.norm(dim=-1, keepdim=True)
                    text_emb = text_emb.squeeze().float()

                # Get CLIP nearest neighbors
                distances, indices = self.index.search(text_emb, from_k)
                matches = self.metadata.get(indices.squeeze().tolist())
                data.append({
                    "query": query,
                    "distances": distances.cpu(),
                    "indices": indices.cpu(),
                    "matches": matches
                })

            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            data_subset = []
            for top_k_query in data:
                data_subset.append({
                    "query": top_k_query["query"],
                    "distances": top_k_query["distances"][:k],
                    "indices": top_k_query["indices"][:k],
                    "matches": top_k_query["matches"][:k]
                })
            return data_subset
