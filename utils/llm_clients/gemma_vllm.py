import time
import random
import requests
from google import genai
from google.genai import types, errors
from .llm_base import BaseLLM
from typing import List, Any, Dict
from openai import OpenAI

import os
os.environ["no_proxy"] = "localhost,127.0.0.1"

class GemmaClient(BaseLLM):
    """Language model that predicts continuation of provided source code using google-genai SDK."""

    def __init__(self, model_name: str, max_output_tokens: int, temperature: float, base_url: str,
                api_key: str, max_retries: int = 5, base_backoff: int = 2, 
                max_backoff: int = 60, num_calls: int = 200, top_p: float = 0.95, 
                min_p: float = 0.0, top_k: int = 64, repetition_penalty: float = 1.0) -> None:
        self.model_name = model_name  # The new SDK handles the "models/" prefix or simple name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        # self.min_p = min_p
        self.top_k = top_k
        # self.repetition_penalty = repetition_penalty
        self.max_retries = max_retries 
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.api_key = api_key
        self.num_calls = num_calls
        
        self.base_url = base_url  # Store the base URL for potential future use, though the SDK handles endpoints

        # Initialize the modern Client
        # self.client = genai.Client(api_key=self.api_key)
        # Initialize the modern Client
        print("Initializing Gemma Client with the new genai SDK...")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        # self.client = genai.Client(api_key=self.api_key)
        print("Gemma Client initialized successfully.")

    def _draw_samples(self, prompt: str, samples_per_prompt: int) -> List[Any]:
        print(f"Generating {samples_per_prompt} samples...")
        all_results = []
        for _ in range(samples_per_prompt):
            sample_data = self._draw_sample(prompt)
            all_results.append(sample_data)
        return all_results

    def _draw_sample(self, prompt: str) -> Dict[str, Any]:
        """Performs a single generation using the new genai client with robust error handling."""

        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                response=self.client.chat.completions.create(
                    model=self.model_name,   # use your served model name
                    messages=[
                            {"role": "user", "content": prompt}
                        ],
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,    
                    top_p=self.top_p,
                )

                generation_time = time.time() - start_time

                text = ""

                # Extract text
                if response.choices:
                    choice = response.choices[0]
                    if choice.message and choice.message.content:
                        text = choice.message.content
                    else:
                        print("Warning: Empty response or blocked output")
                else:
                    print("Warning: No choices returned")

                # Usage metadata extraction
                in_tokens = 0
                out_tokens = 0

                if response.usage:
                    in_tokens = response.usage.prompt_tokens or 0
                    out_tokens = response.usage.completion_tokens or 0
                    print(in_tokens, out_tokens)
                
                # cost_info = self._calculate_cost(self.model_name, in_tokens, out_tokens)
                cost_info = {
                    "input_cost": 0.0,
                    "output_cost": 0.0,
                    "total_cost": 0.0
                } #* gemma runs on free tier over the api
                
                return {
                    "text": text,
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "generation_time": generation_time,
                    **cost_info
                }

            except (errors.ClientError, errors.ServerError) as e:
                # errors.ClientError catches 429 (ResourceExhausted)
                # errors.ServerError catches 500 (Internal) and 503 (Unavailable)
                
                if attempt == self.max_retries:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {e}")
                
                sleep_time = min(self.base_backoff * (2 ** (attempt - 1)), self.max_backoff)
                sleep_time += random.uniform(0, 1)
                
                # Check specifically for Rate Limit (429)
                err_name = "RateLimit" if "429" in str(e) else type(e).__name__
                print(f"Retry {attempt}/{self.max_retries} in {sleep_time:.2f}s due to {err_name}")
                
                time.sleep(sleep_time)
            
            except Exception as e:
                # The "quick accessor" error is gone in the new SDK, but we keep a generic catch
                raise e

    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Calculates cost based on model name and token counts."""
        model = model_id.lower()
        
        # Logic remains the same, though you can now easily distinguish between 
        # flash-lite, flash, and pro via the string name.
        if "pro" in model:
            tier = "small" if input_tokens <= 200_000 else "large"
            in_rate = 1.25 if tier == "small" else 2.50
            out_rate = 10.00 if tier == "small" else 15.00
        elif "flash-lite" in model:
            in_rate, out_rate = 0.10, 0.40
        elif "flash" in model:
            in_rate, out_rate = 0.30, 2.50
        else:
            in_rate, out_rate = 0.0, 0.0

        in_cost = (input_tokens * in_rate) / 1_000_000
        out_cost = (output_tokens * out_rate) / 1_000_000
        
        return {
            "input_cost": round(in_cost, 6),
            "output_cost": round(out_cost, 6),
            "total_cost": round(in_cost + out_cost, 6)
        }