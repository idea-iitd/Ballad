import os
import time
import pickle
from typing import List, Dict, Any
from pathlib import Path
import re

class Logger:
    def __init__(self, problem_name: str, model_name: str, res_dir_name:str, resume_run_id: int = None):
        #create save directory, if it does not exist
        self.results_dir = Path(res_dir_name) / problem_name / model_name
        os.makedirs(self.results_dir, exist_ok=True)

        if resume_run_id is not None:
            # Resume: reuse the existing run_id and load prior logs
            self.run_id = resume_run_id
            self.res_path = os.path.join(
                self.results_dir, f"code_dict_run{resume_run_id}.pkl"
            )
            self.res_dict = self._load_logs()
            print(f"[Logger] Resumed run {resume_run_id} with {len(self.res_dict)} existing entries.")
        else:
            #compute file save name
            self.res_path, self.run_id = self.compute_run_id('code_dict', self.results_dir)
            self.res_dict: List[Dict] = []
        
    def compute_run_id(self, base_name: str, directory: str) -> str:
        # Regex to match exp number
        pattern = re.compile(rf"^{base_name}_run(\d+)\.pkl$")
        existing = [
            f for f in os.listdir(directory)
            if pattern.match(f)
        ]

        if not existing:
            return os.path.join(directory, f"{base_name}_run1.pkl"), 1

        # Extract max exp number
        max_exp = max(int(pattern.match(f).group(1)) for f in existing)
        next_exp = max_exp + 1
        # print(next_exp)
        return os.path.join(directory, f"{base_name}_run{next_exp}.pkl"), next_exp

    def log(
        self,
        call: int,
        parent: str,
        parent_ids: List[str],
        parent_scores: List[float],
        input_prompt_tokens: int,
        generated_program_name: str,
        generated_program_code: Any,
        generated_program_id: str,
        output_tokens: int,
        program_scores: List[float],
        output_gen_time: float,
        eval_time: float,
        timeouts: List[bool],
        best_program_name: str, 
        best_program_code: Any,
        best_program_scores: List[float],
        best_eval_time: float,
        best_func_id: str,
        best_func_timeouts: List[bool],
        gen_model:str,
        best_func_model: str
    ):
        call_data = {
            "call_id": call,
            "parent_prompt": parent,
            "parent_ids": parent_ids,
            "parent_scores": parent_scores,
            "input_prompt_length": input_prompt_tokens,
            "gen_program_name": generated_program_name,
            "gen_program_code": generated_program_code,
            "gen_program_id": generated_program_id,
            "output_tokens": output_tokens,
            "gen_time": output_gen_time,
            "gen_scores": program_scores,
            "gen_model":gen_model,
            "avg_eval_time": eval_time,
            "gen_time_outs": timeouts,
            "best_gen_name": best_program_name,
            "best_gen_code": best_program_code,
            "best_gen_scores": best_program_scores,
            "best_gen_avg_eval_time": best_eval_time,
            "best_gen_id": best_func_id,
            "best_gen_time_outs": best_func_timeouts,
            "best_func_model":best_func_model
        }
        self.res_dict.append(call_data)

    def save_logs(self):
        with open(self.res_path, "wb") as f:
            pickle.dump(self.res_dict, f)
        print(f"Primary log saved to {self.res_path}")

    def _load_logs(self) -> List[Dict]:
        """Load existing log entries from disk if they exist."""
        import pickle
        if os.path.exists(self.res_path):
            with open(self.res_path, 'rb') as f:
                return pickle.load(f)
        return []