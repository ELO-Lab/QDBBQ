import time
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from openelm.configs import ModelConfig

from openai import OpenAI
from google import genai
from google.genai import types

import anthropic
import requests
import json

import logging
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def get_model(config: ModelConfig):
    if config.model_type == "hf":
        return HuggingFaceLLM(config=config)
    elif config.model_type == "openai":
        return OpenAIAPI(config=config, base_url=None)
    elif config.model_type == "grok":
        return OpenAIAPI(config=config, base_url="https://api.x.ai/v1")
    elif config.model_type == "llama":
        return OpenAIAPI(config=config, base_url="https://openrouter.ai/api/v1")
    elif config.model_type == "claude":
        return AnthropicAPI(config=config)
    elif config.model_type == "gemini":
        return GeminiAPI(config=config)
    else:
        raise NotImplementedError



class OpenAIAPI():

    def __init__(self, config, base_url):
            
        self.config: ModelConfig = config
        if base_url is None:
            self.client: Any = OpenAI(
                api_key=config.api_key,
                )
        else:
            self.client: Any = OpenAI(
                api_key=config.api_key,
                base_url=base_url,
                )
        self.tokenizer = None
        self.model = config.model_path
        self.num_output_tokens = config.gen_max_len
        self.temperature = config.temp
        self.max_retries = config.max_retries
        self.type = "commercial"

    def generate(self, prompts: str,  system_prompts: str = None, history: Optional[list[str]] = None):

        messages = []
        if system_prompts is not None:
            messages = [
                {"role": "system", "content": system_prompts},
            ]
        
        if history is not None:
            messages.append({"role": "user", "content": prompts})
            messages.extend(history)
        else:
            messages.append({"role": "user", "content": prompts})
    
        retries = 0
        while retries < self.max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.num_output_tokens,
                    temperature=self.temperature,
                )
                return completion.choices[0].message.content
            
            except Exception as e:
                print(f"API Error: {e}. Retrying in 60 seconds...")
                time.sleep(60)

        raise ValueError
    

class GeminiAPI():

    def __init__(self, config):
            
        self.config: ModelConfig = config
        self.client: Any = genai.Client(
            api_key=config.api_key,
            )
        self.tokenizer = None
        self.model = config.model_path
        self.num_output_tokens = config.gen_max_len
        self.temperature = config.temp
        self.max_retries = config.max_retries
        self.type = "commercial"

    def generate(self, prompts: str,  system_prompts: str = None, history: Optional[list[str]] = None):

        messages = []
        if system_prompts is not None:
            messages = [
                {"role": "system", "content": system_prompts},
            ]
        
        if history is not None:
            messages = prompts + "\n" + history
        else:
            messages = prompts

        retries = 0
        while retries < self.max_retries:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=messages,

                    config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                            maxOutputTokens=self.num_output_tokens,
                            temperature=self.temperature,
                        ),
                    )
                    
                return response.text
            
            except Exception as e:
                print(f"API Error: {e}. Retrying in 15 seconds...")
                time.sleep(15)

        raise ValueError


class AnthropicAPI():

    def __init__(self, config):
            
        self.config: ModelConfig = config
        self.client: Any = anthropic.Anthropic(
            api_key=config.api_key,
        )
        self.tokenizer = None
        self.model = config.model_path
        self.num_output_tokens = config.gen_max_len
        self.temperature = config.temp
        self.max_retries = config.max_retries

    def generate(self, prompts: str,  system_prompts: str = None, history: Optional[list[str]] = None):

        messages = []
        if system_prompts is not None:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompts}]},
            ]
        
        if history is not None:
            messages.extend(history)
            messages.append({"role": "user", "content": [{"type": "text", "text": prompts}]})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": prompts}]})

        retries = 0
        while retries < self.max_retries:
            try:
                message = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.num_output_tokens,
                    temperature=self.temperature,
                )
                    
                return message.content[0].text
            
            except Exception as e:
                print(f"API Error: {e}. Retrying in 60 seconds...")
                time.sleep(60)

        raise ValueError
    


class HuggingFaceLLM():

    def __init__(self, config):
            
        self.config: ModelConfig = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path).to(self.device)
        self.model.eval()
        self.num_output_tokens = config.gen_max_len
        self.temperature = config.temp
        self.max_retries = config.max_retries
        self.type = "open"

    def get_first_token_prob(self, prompt, token_ls):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        next_token_logits = outputs.logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)[0]

        selected_probs = []

        for target in token_ls:
            matching_ids = []

            # Search entire vocabulary for tokens that decode to the target
            # after removing surrounding whitespace.
            for token_id in range(len(self.tokenizer)):
                decoded = self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                if decoded.strip() == target:
                    matching_ids.append(token_id)

            if not matching_ids:
                selected_probs.append(torch.tensor(0.0, device=probs.device))
                continue

            # Sum probability over ALL equivalent variants
            prob = probs[matching_ids].sum()
            selected_probs.append(prob)

        selected_probs = torch.stack(selected_probs)


        normalized_probs = selected_probs / selected_probs.sum()

        return normalized_probs.float().cpu().numpy()


    
    def get_first_token_prob_with_sampling(self, prompt, token_ls, temperature=1.0, top_p=1.0):
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Get logits for NEXT token
        next_token_logits = logits[:, -1, :]  # [1, vocab_size]

        # -------------------------
        # 1. Apply temperature
        # -------------------------
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Convert to probs
        probs = torch.softmax(next_token_logits, dim=-1)  # [1, vocab_size]

        # -------------------------
        # 2. Apply nucleus sampling (top-p)
        # -------------------------
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens outside nucleus
            nucleus_mask = cumulative_probs <= top_p

            # Always keep at least one token
            nucleus_mask[..., 0] = True

            # Zero out tokens outside nucleus
            filtered_probs = torch.zeros_like(probs)
            filtered_probs.scatter_(1, sorted_indices, sorted_probs * nucleus_mask)

            # Renormalize
            probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        # -------------------------
        # 3. Select your tokens
        # -------------------------
        token_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)[0]
            for t in token_ls
        ]

        selected_probs = probs[:, token_ids]  # [1, num_choices]

        # Renormalize over selected tokens
        normalized_probs = selected_probs / selected_probs.sum(dim=-1, keepdim=True)

        return normalized_probs.detach().to(torch.float32).cpu().numpy()[0]