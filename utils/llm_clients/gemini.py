import time
import random
import requests
from google import genai
from google.genai import types, errors
from .llm_base import BaseLLM
from typing import List, Any, Dict

class GeminiLLM(BaseLLM):
    """Language model that predicts continuation of provided source code using google-genai SDK."""

    def __init__(self, model_name: str, max_output_tokens: int, temperature: float, 
                api_key: str, max_retries: int = 5, base_backoff: int = 2, 
                max_backoff: int = 60, num_calls: int = 200) -> None:
        self.model_name = model_name  # The new SDK handles the "models/" prefix or simple name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_retries = max_retries 
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.api_key = api_key
        self.num_calls = num_calls
        
        # Initialize the modern Client
        self.client = genai.Client(api_key=self.api_key)

    def _draw_samples(self, prompt: str, samples_per_prompt: int) -> List[Dict[str, Any]]:
        """Performs a single generation using the new genai client with robust error handling."""
        
        config = types.GenerateContentConfig(
            candidate_count=samples_per_prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
            
                total_generation_time = time.time() - start_time
                per_completion_time = total_generation_time / max(1, len(response.candidates))
                results = []

                for i, cand in enumerate(response.candidates):
                    text = cand.content.parts[0].text if cand.content.parts else ""

                    # Usage metadata extraction
                    if(i == 0):
                        in_tokens = response.usage_metadata.prompt_token_count or 0
                    else:
                        in_tokens = 0

                    out_tokens = self.client.models.count_tokens(model=self.model_name, contents=text).total_tokens if text else 0
                    
                    cost_info = self._calculate_cost(self.model_name, in_tokens, out_tokens)
                    
                    results.append({
                        "text": text,
                        "input_tokens": in_tokens,
                        "output_tokens": out_tokens,
                        "generation_time": per_completion_time,
                        **cost_info
                    })

                return results

            except (errors.ClientError, errors.ServerError) as e:
                # errors.ClientError catches 429 (ResourceExhausted)
                # errors.ServerError catches 500 (Internal) and 503 (Unavailable)
                
                if attempt == self.max_retries:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {e}")
                
                # sleep_time = min(self.base_backoff * (2 ** (attempt - 1)), self.max_backoff)
                sleep_time = 600 # sleep for 10 mins
                # sleep_time += random.uniform(0, 1)
                
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