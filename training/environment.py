import torch
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional

# the grid dimension can be changed, but i'm not really sure how it'll affect performance. the dqn is an mlp, so the grid just gets flattened as input.

# environment
class ChompEnv(gym.Env):
    def __init__(self, GRID_SIZE: Tuple[int, int] = (5, 10), opponent_mode: bool = True) -> None:
        super().__init__() 

        # any fixed fundamental values
        self.GRID_SIZE: Tuple[int, int] = GRID_SIZE
        self.NUM_ROWS: int = GRID_SIZE[0]
        self.NUM_COLS: int = GRID_SIZE[1]
        self.poison: int = (self.NUM_ROWS - 1) * self.NUM_COLS
        
        self.POISON_DIAG_NEIGHBOR_IDX: Tuple[int, int] = (self.NUM_ROWS - 2, 1)
        self.POISON_UP_NEIGHBOR_IDX: Tuple[int, int] = (self.NUM_ROWS - 2, 0)
        self.POISON_RIGHT_NEIGHBOR_IDX: Tuple[int, int] = (self.NUM_ROWS - 1, 1)
        
        # critical moves
        self.poison_right_move: int = (self.POISON_RIGHT_NEIGHBOR_IDX[0] * self.NUM_COLS) + self.POISON_RIGHT_NEIGHBOR_IDX[1]
        self.poison_up_move: int = (self.POISON_UP_NEIGHBOR_IDX[0] * self.NUM_COLS) + self.POISON_UP_NEIGHBOR_IDX[1]
        self.poison_diag_move: int = (self.POISON_DIAG_NEIGHBOR_IDX[0] * self.NUM_COLS) + self.POISON_DIAG_NEIGHBOR_IDX[1]
        
        # initializing spaces
        self.action_space: spaces.Discrete = spaces.Discrete(self.NUM_ROWS * self.NUM_COLS)
        self.observation_space: spaces.Box = spaces.Box(
            low=0,
            high=1,
            shape=self.GRID_SIZE,
            dtype=np.int8
        )
        
        self.opponent_mode: bool = opponent_mode # simulated opponent to train against
        self.grid: torch.Tensor = None
        self.done: bool = False
        
        self.reset()

    # (re)initialize board
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.done = False
        self.grid = torch.ones(self.GRID_SIZE, dtype=torch.int8)
        return self.grid.numpy(), {}

    # current valid action state
    def get_valid_actions(self) -> List[int]:
        edible_indices: torch.Tensor = torch.nonzero(self.grid, as_tuple=False)
        return [((row * self.NUM_COLS) + column) for row, column in edible_indices.tolist()]

    # make the move :3
    def update_grid(self, action: int) -> bool:
        row_index: int = action // self.NUM_COLS
        column_index: int = action % self.NUM_COLS
        poison_eaten: bool = (action == self.poison)

        self.grid[:row_index + 1, column_index:] = 0
        self.done = torch.sum(self.grid).item() == 0

        return poison_eaten

    # oh my god. don't even ask me.
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        truncated: bool = False
        valid_actions: List[int] = self.get_valid_actions()
        reward: float = 0.0
        state: torch.Tensor = self.grid

        # critical states
        poison_diag_empty: bool = bool(state[self.POISON_DIAG_NEIGHBOR_IDX[0], self.POISON_DIAG_NEIGHBOR_IDX[1]] == 0)
        poison_up_empty: bool = bool(state[self.POISON_UP_NEIGHBOR_IDX[0], self.POISON_UP_NEIGHBOR_IDX[1]] == 0)
        poison_right_empty: bool = bool(state[self.POISON_RIGHT_NEIGHBOR_IDX[0], self.POISON_RIGHT_NEIGHBOR_IDX[1]] == 0)

        if action not in valid_actions: # this is not foolproof because i don't think qlearning works like this
            reward = -1000.0
            self.done = True # immediately quit, no new rewards
            return self.grid.numpy(), reward, self.done, truncated, info

        if poison_diag_empty: # this is also not foolproof for the same reason, but the moves i'm restricting basically guarantee loss
            if poison_up_empty and not poison_right_empty:
                action = self.poison_right_move
                reward += 10.0
            elif poison_right_empty and not poison_up_empty:
                action = self.poison_up_move
                reward += 10.0
        else:
            if action == self.poison_right_move or action == self.poison_up_move:
                reward += -50.0
            if action == self.poison_diag_move:
                reward += -50.0

        if not poison_right_empty and not poison_up_empty:
            if action == self.poison_right_move or action == self.poison_up_move:
                reward += -50.0

        if self.done: # redundant but just in case
            return self.grid.numpy(), reward, self.done, truncated, info

        agent_poison: bool = self.update_grid(action) # the agent makes its move
        if agent_poison: # agent loss case
            reward += -550.0
            self.done = True
            return self.grid.numpy(), reward, self.done, truncated, info

        # opponent logic to help train for two player game
        if self.opponent_mode: 
            opponent_valid_actions: List[int] = self.get_valid_actions() # invalid action case
            if not opponent_valid_actions:
                self.done = True
                return self.grid.numpy(), reward, self.done, truncated, info

            if len(opponent_valid_actions) > 1: # pick a random non poison move
                opponent_action: int = random.choice([a for a in opponent_valid_actions if a != self.poison])
            else: # if there is no way to avoid poison, cave
                opponent_action: int = self.poison

            opponent_poison: bool = self.update_grid(opponent_action)
            reward += (150.0 if opponent_poison else 0.0) #reward for the agent if opponent loses
            self.done = self.done or opponent_poison

        return self.grid.numpy(), reward, self.done, truncated, info

    # board render loop
    def render(self) -> str:
        grid: np.ndarray = self.grid.numpy()

        board: str = ""
        header: str = "     " + "  ".join(f"[#D7A7E8]{chr(ord('A') + j)}[/#D7A7E8] " for j in range(self.NUM_COLS))
        board += header + "\n"
        board += "[#C7A6F5]  +" + "----" * self.NUM_COLS + "[/#C7A6F5]\n"

        for i, row in enumerate(grid):
            line: str = f"{i + 1} [#C7A6F5]|[/#C7A6F5]"
            for j, cell in enumerate(row):
                if cell == 1:
                    if i == (self.NUM_ROWS - 1) and j == 0:
                        line += " [#FF80B3]░░[/#FF80B3] "
                    else:
                        line += " [#FFE0F0]██[/#FFE0F0] "
                else:
                    line += "    "
            board += line + "\n"

        return board