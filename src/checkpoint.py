import pickle
from pathlib import Path

class CheckpointManager:
    """Saves and loads full experiment state."""

    def __init__(self, results_dir: Path, run_id: int):
        self.path = results_dir / "checkpoint" / f"checkpoint_run{run_id}.pkl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, database, router, sampler_state: dict) -> None:
        state = {'sampler': sampler_state}
        # Database and router save themselves to separate files
        database.save_checkpoint(str(self.path.with_suffix('.db.pkl')))
        if router is not None:
            router.save_checkpoint(str(self.path.with_suffix('.router.pkl')))
        with open(self.path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[Checkpoint] Saved at {self.path}")

    def load(self) -> dict:
        with open(self.path, 'rb') as f:
            return pickle.load(f)

    @property
    def db_path(self) -> str:
        return str(self.path.with_suffix('.db.pkl'))

    @property
    def router_path(self) -> str:
        return str(self.path.with_suffix('.router.pkl'))