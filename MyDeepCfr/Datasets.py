
import random
from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader
import ray



class AdvantageDataset(Dataset):
    def __init__(self, max_capacity: int):
        self._buffer = deque(maxlen=max_capacity)

    def add(self, information_state, iteration, samp_regret, legal_action_mask):
        self._buffer.append((information_state, iteration, samp_regret, legal_action_mask))

    def sample(self, num_samples):
        if len(self._buffer) < num_samples:
            raise ValueError(f'{num_samples} elements could not be sampled from size {len(self._buffer)}')
        return random.sample(self._buffer, num_samples)

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, idx):
        return self._buffer[idx]

    def shuffle_data(self):
        random.shuffle(self._buffer)

    def get_dataloader(self, batch_size: int, shuffle: bool = True):
        info_states = torch.tensor([x[0] for x in self._buffer], dtype=torch.float32)
        iterations = torch.tensor([x[1] for x in self._buffer], dtype=torch.float32)
        samp_regrets = torch.tensor([x[2] for x in self._buffer], dtype=torch.float32)
        legal_actions = torch.tensor([x[3] for x in self._buffer], dtype=torch.float32)

 
        dataset = torch.utils.data.TensorDataset(info_states, iterations, samp_regrets, legal_actions)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,  # Параллельная загрузка данных
            drop_last=True,  # Игнорируем последний батч, если он меньше batch_size
        )
        return dataloader

class  StrategyDataset(Dataset):
    def __init__(self,max_capacity:int):
        self._buffer=deque(maxlen=max_capacity)

        
    def add(self,information_state,iteration,strategy,legal_action_mask):
        self._buffer.append((information_state,iteration,strategy,legal_action_mask))

    def sample(self, num_samples):
            if len(self._buffer)< num_samples:
                raise ValueError('{} elements could not be sampled from size {}'.format(
                    num_samples,  len(self._buffer)))
            return random.sample(self._buffer, num_samples)
    
    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    @property
    def buffer(self):
        return self._buffer

    def shuffle_data(self):
        random.shuffle(self._buffer)

    def get_dataloader(self, batch_size: int, shuffle: bool = True):
        info_states = torch.tensor([x[0] for x in self._buffer], dtype=torch.float32)
        iterations = torch.tensor([x[1] for x in self._buffer], dtype=torch.float32)
        strategies = torch.tensor([x[2] for x in self._buffer], dtype=torch.float32)
        legal_actions = torch.tensor([x[3] for x in self._buffer], dtype=torch.float32)


        dataset = torch.utils.data.TensorDataset(info_states, iterations, strategies, legal_actions)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Параллельная загрузка данных
            drop_last=True,  # Игнорируем последний батч, если он меньше batch_size
        )
        return dataloader