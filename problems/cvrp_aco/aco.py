import torch
from torch.distributions import Categorical
import random
import itertools
import numpy as np


class BatchedACO:
    """Runs B CVRP instances simultaneously using batched (B,n,n) tensors on GPU/MPS/CPU."""

    def __init__(self, distances, demand, heuristic, capacity,
                 n_ants=30, decay=0.9, alpha=1, beta=1, device='cpu'):
        # distances:  (B, n, n)
        # demand:     (B, n)
        # heuristic:  (B, n, n)
        self.batch_size   = distances.shape[0]
        self.problem_size = distances.shape[1]
        self.distances  = distances
        self.demand     = demand
        self.capacity   = capacity
        self.n_ants     = n_ants
        self.decay      = decay
        self.alpha      = alpha
        self.beta       = beta
        self.device     = device

        self.pheromone   = torch.ones_like(distances)
        self.heuristic   = heuristic
        self.lowest_cost = torch.full((self.batch_size,), float('inf'), device=device)

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            paths = self.gen_path()              # (seq_len, B, n_ants)
            costs = self.gen_path_costs(paths)   # (B, n_ants)
            best_costs, _ = costs.min(dim=-1)    # (B,)
            improved = best_costs < self.lowest_cost
            self.lowest_cost[improved] = best_costs[improved]
            self.update_pheromone(paths, costs)
        return self.lowest_cost                  # (B,)

    @torch.no_grad()
    def update_pheromone(self, paths, costs):
        self.pheromone.mul_(self.decay)
        seq_len, B, n_ants = paths.shape
        n = self.problem_size

        src = paths[:-1]  # (seq_len-1, B, n_ants)
        dst = paths[1:]

        b_idx = (torch.arange(B, device=self.device)
                 .view(1, B, 1).expand(seq_len - 1, B, n_ants))

        updates = (1.0 / costs).unsqueeze(0).expand(seq_len - 1, B, n_ants)

        flat_idx = (b_idx * n * n + src * n + dst).reshape(-1)
        flat_upd = updates.reshape(-1)

        self.pheromone.view(-1).scatter_add_(0, flat_idx, flat_upd)
        self.pheromone.clamp_(min=1e-10)

    @torch.no_grad()
    def gen_path_costs(self, paths):
        # paths: (seq_len, B, n_ants)
        u = paths.permute(1, 2, 0)   # (B, n_ants, seq_len)
        v = torch.roll(u, shifts=-1, dims=-1)
        b_idx = torch.arange(self.batch_size, device=self.device).view(-1, 1, 1)
        # distances[b, u[b,a,t], v[b,a,t]], summed over t (excluding last wrap-around step)
        return self.distances[b_idx, u[:, :, :-1], v[:, :, :-1]].sum(dim=-1)  # (B, n_ants)

    def gen_path(self):
        B, n_ants, n = self.batch_size, self.n_ants, self.problem_size
        actions      = torch.zeros(B, n_ants, dtype=torch.long, device=self.device)
        visit_mask   = torch.ones(B, n_ants, n, device=self.device)
        visit_mask   = self._update_visit_mask(visit_mask, actions)
        used_cap     = torch.zeros(B, n_ants, device=self.device)
        used_cap, cap_mask = self._update_capacity_mask(actions, used_cap)

        paths_list = [actions]
        done = self._check_done(visit_mask, actions)
        while not done:
            actions    = self._pick_move(actions, visit_mask, cap_mask)
            paths_list.append(actions)
            visit_mask = self._update_visit_mask(visit_mask, actions)
            used_cap, cap_mask = self._update_capacity_mask(actions, used_cap)
            done = self._check_done(visit_mask, actions)

        return torch.stack(paths_list)  # (seq_len, B, n_ants)

    def _pick_move(self, prev, visit_mask, capacity_mask):
        B, n_ants = prev.shape
        b_idx     = torch.arange(B, device=self.device).unsqueeze(1)
        pheromone = self.pheromone[b_idx, prev, :]   # (B, n_ants, n)
        heuristic = self.heuristic[b_idx, prev, :]   # (B, n_ants, n)
        dist      = (pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask * capacity_mask
        return Categorical(dist.view(B * n_ants, -1)).sample().view(B, n_ants)

    def _update_visit_mask(self, visit_mask, actions):
        B, n_ants = actions.shape
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, n_ants)
        a_idx = torch.arange(n_ants, device=self.device).unsqueeze(0).expand(B, n_ants)
        visit_mask[b_idx, a_idx, actions] = 0
        visit_mask[:, :, 0] = 1  # depot always revisitable
        at_depot      = (actions == 0)
        has_unvisited = (visit_mask[:, :, 1:] != 0).any(dim=-1)
        visit_mask[:, :, 0][at_depot & has_unvisited] = 0
        return visit_mask

    def _update_capacity_mask(self, cur_nodes, used_capacity):
        B = self.batch_size
        capacity_mask = torch.ones(B, self.n_ants, self.problem_size, device=self.device)
        used_capacity[cur_nodes == 0] = 0
        b_idx = torch.arange(B, device=self.device).unsqueeze(1)
        used_capacity = used_capacity + self.demand[b_idx, cur_nodes]  # (B, n_ants)
        remaining  = (self.capacity - used_capacity).unsqueeze(-1)     # (B, n_ants, 1)
        demand_exp = self.demand.unsqueeze(1)                          # (B, 1, n)
        capacity_mask[demand_exp > remaining] = 0
        return used_capacity, capacity_mask

    def _check_done(self, visit_mask, actions):
        return (visit_mask[:, :, 1:] == 0).all() and (actions == 0).all()


class ACO():

    def __init__(self,  # 0: depot
                 distances, # (n, n)
                 demand,   # (n, )
                 heuristic, # (n, n)
                 capacity,
                 n_ants=30, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu',
                 ):
        
        self.problem_size = len(distances)
        self.distances = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
        self.demand = torch.tensor(demand, device=device) if not isinstance(demand, torch.Tensor) else demand
        self.capacity = capacity
                
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.pheromone = torch.ones_like(self.distances)
        self.heuristic = torch.tensor(heuristic, device=device) if not isinstance(heuristic, torch.Tensor) else heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device
        

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            paths = self.gen_path()
            costs = self.gen_path_costs(paths)
            
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost
       
            self.update_pheronome(paths, costs)

        return self.lowest_cost
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        for i in range(self.n_ants):
            path = paths[:, i]
            cost = costs[i]
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost
        self.pheromone[self.pheromone < 1e-10] = 1e-10
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        u = paths.permute(1, 0) # shape: (n_ants, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)  
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self):
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)
        
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        
        paths_list = [actions] # paths_list[i] is the ith move (tensor) for all ants
        
        done = self.check_done(visit_mask, actions)
        while not done:
            actions = self.pick_move(actions, visit_mask, capacity_mask)
            paths_list.append(actions)
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            done = self.check_done(visit_mask, actions)
            
        return torch.stack(paths_list)
        
    def pick_move(self, prev, visit_mask, capacity_mask):
        pheromone = self.pheromone[prev] # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev] # shape: (n_ants, p_size)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask * capacity_mask) # shape: (n_ants, p_size)
        dist = Categorical(dist)
        actions = dist.sample() # shape: (n_ants,)
        return actions
    
    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask
    
    def update_capacity_mask(self, cur_nodes, used_capacity):
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size) # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants, 1) # (n_ants, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0
        
        return used_capacity, capacity_mask
    
    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()    

    