from typing import List, Any

class BaseLLM:
    """Base class for any LLM adapter."""
    
    def _draw_samples(self, prompt: str, samples_per_prompt: int) -> List[Any]:
        """
        Returns a list of completions:
        Each tuple: (text, prompt_tokens, completion_tokens, generation_time)
        """
        raise NotImplementedError("Subclasses must implement this method")
