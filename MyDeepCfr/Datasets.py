
from collections import deque
import random

class  AdvantageDataset:
    def __init__(self,max_capacity:int):
        self._buffer=deque(maxlen=max_capacity)

        
    def add(self,information_state,iteration,samp_regret,legal_action_mask):
        self._buffer.append((information_state,iteration,samp_regret,legal_action_mask))

    def sample(self, num_samples):
            if len(self._buffer)< num_samples:
                raise ValueError('{} elements could not be sampled from size {}'.format(
                    num_samples,  len(self._buffer)))
            return random.sample(self._buffer, num_samples)
    
    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return self._buffer.count()

    def __iter__(self):
        return iter(self._buffer)

    @property
    def buffer(self):
        return self._buffer

    def shuffle_data(self):
        random.shuffle(self._buffer)


class  StrategyDataset:
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
        return self._buffer.count()

    def __iter__(self):
        return iter(self._buffer)

    @property
    def buffer(self):
        return self._buffer

    def shuffle_data(self):
        random.shuffle(self._buffer)
