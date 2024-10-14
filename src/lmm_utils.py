import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import requests


# Make sure your transformers package is new enough
# pip install tranformers==4.40.1
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    AutoProcessor, LlavaForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    AutoModelForCausalLM, AutoTokenizer,
    AutoProcessor, PaliGemmaForConditionalGeneration
)


def convert_gpt_response_to_preds(response, warn=False):
    preds = []
    for i in range(len(response)):
        if type(response[i]) == requests.models.Response and response[i].status_code == 400:
            pred = 0
        else:
            top_logprobs = response[i]['choices'][0]['logprobs']['content'][0]['top_logprobs']
            yes_logprobs = [lp['logprob'] for lp in top_logprobs if lp['token'] == 'Yes']
            no_logprobs = [lp['logprob'] for lp in top_logprobs if lp['token'] == 'No']

            if len(yes_logprobs) == 0:
                pred = 0
            elif len(no_logprobs) == 0:
                pred = 1
            else:
                assert len(no_logprobs)  >= 1
                yes_prob = np.exp(yes_logprobs).sum()
                no_prob = np.exp(no_logprobs).sum()
                pred = yes_prob / (yes_prob + no_prob)
                if yes_prob + no_prob < 0.9 and warn:
                    print('bad output tokens with values:', yes_prob, no_prob)
        preds.append(pred)
    y_pred = np.asarray(preds)
    return y_pred


class ModelWrapper:
    """This class provides a neat wrapper around different LVLM models to help evaluate
    their scores to Yes/No questions."""
    
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        
        self.load_model(model_name, device)
    
    def load_model(self, model_name, device):
        print("Loading model:", model_name)

        if model_name.startswith('Salesforce/blip2-'):
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)
            
            self.yes_token = self.processor.tokenizer("Yes").input_ids[0]
            self.no_token = self.processor.tokenizer("No").input_ids[0]
            
        elif model_name.startswith('Salesforce/instructblip'):

            self.model = InstructBlipForConditionalGeneration.from_pretrained(model_name).to(device)
            self.processor = InstructBlipProcessor.from_pretrained(model_name)
            
            self.yes_token = self.processor.tokenizer("Yes").input_ids[0]
            self.no_token = self.processor.tokenizer("No").input_ids[0]
            
        elif model_name.startswith('llava-hf/llava-1.5'):
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                use_flash_attention_2=True
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            self.yes_token = self.processor.tokenizer("Yes").input_ids[1]
            self.no_token = self.processor.tokenizer("No").input_ids[1]
            
            
        elif model_name.startswith('llava-hf/llava-v1.6-mistral'):
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                use_flash_attention_2=True
            ).to(device) 
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            
            self.yes_token = self.processor.tokenizer("Yes").input_ids[1]
            self.no_token = self.processor.tokenizer("No").input_ids[1]
#             self.yes_token = self.processor.tokenizer("\nYes").input_ids[-1]
#             self.no_token = self.processor.tokenizer("\nNo").input_ids[-1]
            
        elif model_name.startswith('llava-hf/llava-v1.6-34b'):
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                use_flash_attention_2=True
            ).to(device) 
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            
            self.yes_token = self.processor.tokenizer(" Yes").input_ids[1]
            self.no_token = self.processor.tokenizer(" No").input_ids[1]
            
        elif model_name.startswith('google/paligemma'):
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device
            ).to(device).eval()
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            self.yes_token = self.processor.tokenizer("yes").input_ids[-1]
            self.no_token = self.processor.tokenizer("no").input_ids[-1]

        else:
            raise ValueError(f"Model with name '{model_name}' not found.")

        print(f"    'Yes' token: {self.yes_token}")
        print(f"    'No' token:  {self.no_token}")
    
    def scores_to_pred(self, scores):
        """Converts output logits to a binary prediction score."""
        probs = scores.exp()
        prob_yes = probs[self.yes_token]
        prob_no = probs[self.no_token]

        if prob_yes + prob_no < 0.9:
            print('[warn] got bad output tokens')

        score = (prob_yes) / (prob_yes + prob_no)
        return score.item()
    
    def generate(self, image_name, prompt, raw_image=None, data_folder=None, temperature=0.5, output_scores=True, **generation_args):
        if raw_image is None and data_folder is None:
            assert raw_image is not None, 'You must provide either the data_folder the load the image from, or the raw_image to use'
        
        if raw_image is None:
            image_file = os.path.join(data_folder, image_name)
            raw_image = Image.open(image_file)
        
        # BLIP models require image conversion BGR->RGB (https://huggingface.co/Salesforce/instructblip-flan-t5-xxl)
        if self.model_name.startswith('Salesforce/blip2-') or self.model_name.startswith('Salesforce/instructblip'):
            raw_image = raw_image.convert('RGB')
        
#         pad_token_id = None
#         if model_name.startswith('llava-hf/llava-v1.6') or model_name.startswith("Salesforce/instructblip") or model_name.startswith('Salesforce/blip2'):
        pad_token_id = self.processor.tokenizer.eos_token_id
        
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, 
                                  do_sample=True if temperature > 0 else False,
                                  temperature=temperature,
                                  max_new_tokens=5,
                                  use_cache=True,
                                  return_dict_in_generate=output_scores,
                                  output_scores=output_scores,
                                  renormalize_logits=True,
                                  pad_token_id=pad_token_id,
                                  **generation_args)
        
        return out
    
    def score_image(self, image_name, prompt, data_folder=None, raw_image=None, **generation_args):
        out = self.generate(image_name, prompt, raw_image=raw_image, data_folder=data_folder, **generation_args)
        logits = out.scores[0][0]
        return self.scores_to_pred(logits)
    
    def score_images(self, images, prompt, data_folder, **generation_args):
        scores = []
        for image_name in images:
            scores.append(self.score_image(image_name, prompt, data_folder, **generation_args))
        return np.asarray(scores)
        
        
class ModelWrapperWithCache(ModelWrapper):
    """Wraps ModelWrapper with a cache that saves previous results to save computation."""
    
    def __init__(self, model_name, device, cache_dir='cache', load_cache_only=False):
        self.load_cache_only = load_cache_only
        self.model_name = model_name
        self.device = device
        self.model = None
        
        model_name_sanitized = model_name.replace(":", "__").replace("/", "--")
        if model_name.startswith("openai"):
            self.cache_path = os.path.join(cache_dir, f"{model_name_sanitized}--nosystem-cache.npy")
        else:
            self.cache_path = os.path.join(cache_dir, f"{model_name_sanitized}--cache.npy")
        # print('cache path:', self.cache_path)
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = np.load(f, allow_pickle=True).item()
#             print("Loaded cache with", len(self.cache), "items.")
        else:
            print(f"No cache found at {self.cache_path}. Initialized empty.")
            self.cache = {}
            
    def save(self):
        assert type(self.cache) == dict
        np.save(self.cache_path, self.cache)
        
    def score_images(self, images, prompt, data_folder=None, raw_images=None, temperature=0.5, override_cache=False, **generation_args):
        scores = []
        for idx, image_name in enumerate(images):
            key = image_name + "--" + prompt
            if key in self.cache and not override_cache:
                score = self.cache[key]
            elif not self.load_cache_only:
                if self.model is None:
                    print("Prediction not cached, loading model...")
                    self.load_model(self.model_name, self.device)
                raw_image = None if raw_images is None else raw_images[idx]
                score = self.score_image(image_name, prompt, data_folder=data_folder, raw_image=raw_image, temperature=0.5, **generation_args)
                self.cache[key] = score
            else:
                raise ValueError("Tried to load from cache, but no entry found. If you mean to load "
                                 "the model, run with load_cache_only=True")
            scores.append(score)
        return np.asarray(scores)


def get_lmm_prompt(query, model_name):
    if model_name.startswith("Salesforce/instructblip") or model_name.startswith("Salesforce/blip2-"):
        prompt = (f"Does this picture show {query}?"
                   "Answer the question with either \"Yes\" or \"No\" and nothing else."
                   "\nAnswer: ")
    elif model_name.startswith("llava-hf/llava-1.5"):
        prompt = (f"USER: <image>\nDoes this picture show {query}?"
                  "Answer the question with either \"Yes\" or \"No\" and nothing else."
                  "\nASSISTANT:")
    elif model_name.startswith("llava-hf/llava-v1.6-mistral"):
        prompt = (f"[INST] <image>\nDoes this picture show {query}?"
                  "Answer the question with either \"Yes\" or \"No\" and nothing else. [/INST]")
    elif model_name == "llava-hf/llava-v1.6-34b-hf":
         prompt = ("<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n"
              f"Does this picture show \"{query}\"?"
              "Answer the question with either \"Yes\" or \"No\".<|im_end|><|im_start|>answer\n")
    elif model_name.startswith("google/paligemma"):
        prompt = (f"Q: Does this picture show {query}? Respond with yes or no.\n"
                       "A:")
    elif model_name.startswith('Efficient-Large-Model/VILA'):
        prompt = (f"<image>\n Does this picture show {query}?\n"
                  "Answer the question with either \"Yes\" or \"No\" and nothing else.")
    elif model_name.startswith('openai-gpt'):
        prompt = (f"Does this picture show exactly \"{query}\"?\n"
                   "Answer the question with either \"Yes\" or \"No\" and nothing else.")
    else:
        raise ValueError(f'model name {model_name} does not match to a known prompt format')
    return prompt