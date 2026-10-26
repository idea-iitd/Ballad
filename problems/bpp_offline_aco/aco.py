import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Annotated
from math import floor
import torch
from torch.distributions import Categorical


def _get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class BatchedACO:
    """
    Batched BPP ACO: runs B instances × A ants simultaneously using PyTorch tensors.

    All instances must have the same n (item count) and capacity, which holds for
    any single dataset file (val500, test500, etc.).
    """

    def __init__(self, demand, heuristic, capacity,
                 n_ants=20, decay=0.95, alpha=1, beta=1, device='cpu'):
        # demand:    (B, n) int32 tensor — item demands per instance
        # heuristic: (B, n, n) float32 — pre-clipped and normalized per instance
        # capacity:  int (same for all instances, =150)
        self.batch_size   = demand.shape[0]
        self.problem_size = demand.shape[1]
        self.capacity = capacity
        self.n_ants   = n_ants
        self.decay    = decay
        self.alpha    = alpha
        self.beta     = beta
        self.device   = device

        self.demand    = demand      # (B, n)
        self.heuristic = heuristic  # (B, n, n)
        self.pheromone = torch.ones(
            self.batch_size, self.problem_size, self.problem_size,
            dtype=torch.float32, device=device
        )
        self.best_cost = torch.full((self.batch_size,), float(self.problem_size), device=device)

    @torch.no_grad()
    def run(self, n_iterations: int) -> torch.Tensor:
        for _ in range(n_iterations):
            prob = self.pheromone ** self.alpha * self.heuristic ** self.beta  # (B, n, n)
            paths, costs, fitnesses = self._gen_paths(self.n_ants, prob)
            # paths: (B, A, n)  costs: (B, A)  fitnesses: (B, A)

            best_costs, _ = costs.min(dim=-1)       # (B,)
            improved = best_costs < self.best_cost
            self.best_cost = torch.where(improved, best_costs.float(), self.best_cost)

            self._update_pheromone(paths, fitnesses)

        return self.best_cost   # (B,)  — number of bins per instance

    @torch.no_grad()
    def _update_pheromone(self, paths, fitnesses):
        # paths: (B, A, n) long,  fitnesses: (B, A) float
        B, A, n = paths.shape
        self.pheromone.mul_(self.decay)
        # Process one ant at a time to keep (B,n,n) memory footprint
        for a in range(A):
            path_a = paths[:, a, :]          # (B, n)
            f_a    = fitnesses[:, a]          # (B,)
            same_bin = (path_a.unsqueeze(-1) == path_a.unsqueeze(-2))  # (B, n, n)
            self.pheromone.add_(same_bin.float() * f_a.view(B, 1, 1) / A)

    def _gen_paths(self, n_ants: int, prob: torch.Tensor):
        """Generate paths for all ants across all instances (batched sequential placement)."""
        B, n, A = self.batch_size, self.problem_size, n_ants
        d = self.device

        # Per-(B,A) state
        path         = torch.full((B, A, n), -1,    dtype=torch.long,  device=d)
        valid_items  = torch.ones(B, A, n,           dtype=torch.bool,  device=d)
        current_bin  = torch.zeros(B, A,             dtype=torch.long,  device=d)
        bin_vacancy  = torch.full((B, A), self.capacity, dtype=torch.long, device=d)
        bin_items    = torch.zeros(B, A, n,           dtype=torch.bool,  device=d)
        item_count   = torch.zeros(B, A,             dtype=torch.long,  device=d)
        occ_sq_sum   = torch.zeros(B, A,             dtype=torch.float32, device=d)

        demand_exp = self.demand.unsqueeze(1).expand(B, A, n)  # (B, A, n) — static
        b_idx = torch.arange(B, device=d).unsqueeze(1).expand(B, A)
        a_idx = torch.arange(A, device=d).unsqueeze(0).expand(B, A)

        for _ in range(n):
            # --- Determine which ants need a new bin ---
            bin_vac_exp  = bin_vacancy.unsqueeze(-1).expand(B, A, n)
            mask         = (demand_exp <= bin_vac_exp) & valid_items     # (B, A, n)
            needs_new    = ~mask.any(dim=-1)                             # (B, A)

            # Close current bin: accumulate occupied^2 contribution
            occ = (self.capacity - bin_vacancy).float()
            occ_sq_sum = occ_sq_sum + (occ / self.capacity) ** 2 * needs_new.float()

            # Open new bin: reset vacancy, bin_items, item_count
            bin_vacancy = torch.where(needs_new, torch.full_like(bin_vacancy, self.capacity), bin_vacancy)
            current_bin = current_bin + needs_new.long()
            bin_items   = bin_items & ~needs_new.unsqueeze(-1)
            item_count  = item_count * (~needs_new).long()

            # Recompute mask with updated vacancy
            bin_vac_exp = bin_vacancy.unsqueeze(-1).expand(B, A, n)
            mask        = (demand_exp <= bin_vac_exp) & valid_items      # (B, A, n)

            # --- Compute selection probabilities ---
            # Case A: needs_new_bin  → uniform over all remaining valid items
            p_new = valid_items.float() + 1e-10

            # Case B: first item in bin (item_count==0, bin already open)
            p_first = mask.float() + 1e-10

            # Case C: subsequent items → prob-weighted by current bin occupants
            # prob_sum[b,a,i] = sum_j prob[b,j,i] * bin_items[b,a,j]
            prob_sum = torch.einsum('bji,baj->bai', prob, bin_items.float())  # (B, A, n)
            p_weighted = (prob_sum / (item_count.float().unsqueeze(-1) + 1e-5) + 1e-5) * mask.float()

            is_first  = (item_count == 0) & ~needs_new
            final_prob = torch.where(
                needs_new.unsqueeze(-1), p_new,
                torch.where(is_first.unsqueeze(-1), p_first, p_weighted)
            ).clamp(min=1e-10)

            # --- Sample item ---
            sampled = Categorical(final_prob.view(B * A, n)).sample().view(B, A)  # (B, A)

            # --- Update state ---
            bin_items[b_idx, a_idx, sampled]   = True
            bin_vacancy                         = bin_vacancy - self.demand[b_idx, sampled]
            valid_items[b_idx, a_idx, sampled]  = False
            path[b_idx, a_idx, sampled]         = current_bin
            item_count                          = item_count + 1

        # Add final open bin contribution
        occ_final  = (self.capacity - bin_vacancy).float()
        occ_sq_sum = occ_sq_sum + (occ_final / self.capacity) ** 2

        costs     = (current_bin + 1).float()              # (B, A) bins used
        fitnesses = occ_sq_sum / costs                     # (B, A) mean occ^2

        return path, costs.long(), fitnesses

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

def organize_path(path: IntArray) -> Tuple[int, IntArray]:
    order = {}
    result = np.zeros_like(path)
    for i, v in enumerate(path):
        if v in order:
            result[i] = order[v]
        else:
            result[i] = order[v] = len(order)
    return len(order), result

def calculate_path_cost_fitness(vacancies: IntArray, capacity: int) -> Tuple[int, float]:
    occupied = (capacity - vacancies[vacancies!=capacity]).astype(float)
    cost = len(occupied)
    result = ((occupied/capacity)**2).sum().item()/cost
    return cost, result

def calculate_path_fitness(vacancies: List[int], capacity: int) -> float:
    occupied = capacity - np.array(vacancies, dtype=float)
    result = ((occupied/capacity)**2).sum().item()/len(vacancies)
    return result

def greedy_sample(prob: FloatArray) -> int:
    return prob.argmax().item()

def random_sample(prob: FloatArray) -> int:
    # not used, `random_sample_discrete_distribution` is a faster implementation
    sampled = np.random.choice(prob.size, p=prob/prob.sum())
    return sampled

def random_sample_discrete_distribution(prob: FloatArray) -> int:
    # prob_exp = np.exp(prob-prob.max())
    # prob_exp[prob==0] = 0
    # np.random.choice is somehow slow
    cumprob = np.cumsum(prob)
    sampled = np.searchsorted(cumprob, next(uniform_generator)*cumprob[-1]).item()
    return sampled if sampled<len(cumprob) else len(cumprob)-1

def uniform_number_generator(batch_size = 500):
    # it's also slow to generate random numbers one by one
    while 1:
        numbers = np.random.random(batch_size)
        for n in numbers:
            yield n.item()

uniform_generator = uniform_number_generator()


def reset_uniform_generator() -> None:
    """Discard the current pre-fetched buffer and start a fresh generator.

    Call this after np.random.seed() so the buffer is filled from the
    newly-seeded RNG rather than from a stale pre-seeding batch.
    """
    global uniform_generator
    uniform_generator = uniform_number_generator()


class ACO(object):
    def __init__(self,
                 demand: IntArray,   # (n, )
                 heuristic: FloatArray,   # (n, n)
                 capacity: int,
                 n_ants=20, 
                 decay=0.95,
                 alpha=1,
                 beta=1,
                 greedy = False
                 ):
        
        self.problem_size = len(demand)
        self.capacity = capacity
        self.demand = demand
        assert self.demand.max() <= self.capacity
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.pheromone: FloatArray = np.ones((self.problem_size, self.problem_size)) # problem_size x self.problem_size
        heuristic[heuristic > 1e6] = 1e6
        heuristic[heuristic < 1e-6] = 1e-6
        heuristic = heuristic/heuristic.max() # normalize
        heuristic[heuristic < 1e-6] = 1e-6
        self.heuristic: FloatArray = heuristic # problem_size x self.problem_size

        self.shortest_path: IntArray = np.arange(self.problem_size)
        self.best_cost = self.problem_size

        self._ordinal: IntArray = np.arange(self.problem_size, dtype=int) # for indexing
        self.greedy_mode = greedy
    
    def run(self, iterations: int) -> Tuple[int, IntArray]:
        for _ in range(iterations):
            prob = self.pheromone**self.alpha * self.heuristic**self.beta
            paths, costs, fitnesses = self.gen_paths(self.n_ants, prob)
            best_index = costs.argmin()
            best_cost = costs[best_index].item()
            if best_cost < self.best_cost:
                self.shortest_path = paths[best_index]
                self.best_cost = best_cost
            self.update_pheronome(paths, fitnesses)
        assert self.is_valid_path(self.shortest_path)
        # cost, path = organize_path(self.shortest_path)
        # assert cost >= np.ceil(np.sum(self.demand).astype(float)/self.capacity).item()
        return organize_path(self.shortest_path)

    def sample_only(self, count: int) -> Tuple[int, IntArray]:
        self.greedy_mode = True
        paths, costs, _ = self.gen_paths(count, self.heuristic)
        best_index = costs.argmin()
        best_path = paths[best_index]
        assert self.is_valid_path(best_path)
        return organize_path(best_path)

    def update_pheronome(self, paths: List[IntArray], fitnesses: FloatArray):
        delta_phe = np.zeros_like(self.pheromone) # problem_size x problem_size
        for path, f in zip(paths, fitnesses):
            delta_phe[path[:, None]==path[None, :]] += f / self.n_ants
        self.pheromone *= self.decay
        self.pheromone += delta_phe

    def gen_paths(self, count: int, prob: FloatArray) -> Tuple[List[IntArray], IntArray, FloatArray]:
        paths, costs, fitnesses = [], [], []
        for _ in range(count):
            path, cost, fitness = self.sample_path(prob)
            paths.append(path)
            costs.append(cost)
            fitnesses.append(fitness)
        return paths, np.array(costs, dtype=int), np.array(fitnesses, dtype=float)
    
    def sample_path(self, prob: FloatArray
                    ) -> Tuple[
                        Annotated[IntArray, "sampled path"], 
                        Annotated[int, "used bins"], 
                        Annotated[float, "fitness"]]:
        
        if self.greedy_mode:
            sample_func = greedy_sample
        else:
            sample_func = random_sample_discrete_distribution
    
        path = np.ones(self.problem_size, dtype=int)*-1 # x=path[i] => put item i in bin x
        valid_items = np.ones(self.problem_size, dtype=bool)
        current_bin = item_count = 0
        vacancies = []
        bin_vacancy = self.capacity
        bin_items = np.zeros_like(valid_items)

        for _ in range(self.problem_size):
            mask = np.bitwise_and(self.demand <= bin_vacancy, valid_items)
            if not np.any(mask): # no valid item
                # move to the next bin
                vacancies.append(bin_vacancy)
                bin_vacancy, item_count = self.capacity, 0
                current_bin += 1
                bin_items[:] = False
                # uniformly select one
                selected = self.random_select(valid_items)
            else:
                if item_count == 0:
                    selected = self.random_select(mask)
                else:
                    item_prob = (prob[bin_items].sum(0)/item_count+1e-5) * mask
                    selected = sample_func(item_prob)
            
            # put item in this bin
            bin_items[selected] = True
            bin_vacancy -= self.demand[selected]
            valid_items[selected] = False
            path[selected] = current_bin
            item_count += 1
        
        vacancies.append(bin_vacancy)
        fitness = calculate_path_fitness(vacancies, self.capacity)
        return path, len(vacancies), fitness
    
    def random_select(self, mask: npt.NDArray[np.bool_]) -> int:
        valid = self._ordinal[mask]
        return valid[floor(next(uniform_generator)*len(valid))].item()
        # return valid[np.random.randint(0, len(valid))].item()
    
    def is_valid_path(self, path: IntArray) -> bool:
        # not used
        if path.shape[0] != self.problem_size:
            return False
        bins, path = organize_path(path)
        occupied = np.zeros(bins, dtype=int)
        for i, v in enumerate(path):
            if v<0:
                return False
            occupied[v] += self.demand[i]
            if occupied[v] > self.capacity:
                return False
        return True

if __name__ == '__main__':
    np.random.seed(1)
    n = 10
    demand = np.random.randint(5, 15, size=(n,))
    np.random.seed()
    print(*demand)
    initheu = np.tile(demand/demand.max(), (demand.shape[0], 1))
    aco = ACO(demand, initheu, capacity=25)
    # print(*aco.sample_path(aco.heuristic))
    print(*aco.run(50))

