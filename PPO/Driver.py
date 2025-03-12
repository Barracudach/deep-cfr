import random
import torch
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase
from collections import namedtuple
from .TrajectoryManager import TrajectoryManager
from .Networks.Actor import Actor


class PPODriver(WorkerBase):
    def  __init__(self,actions):
        super().__init__(t_prof=namedtuple('profile', ['DISTRIBUTED','CLUSTER'])(DISTRIBUTED=True,CLUSTER=False))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Launching on GPU")
        else:
            self.device = torch.device("cpu")
            print("Launching on CPU")

        self.trajectory_manager=TrajectoryManager(actions)
        self.actor=Actor(len(actions)+2).to(self.device)


    def run(self):
        for i in range(100000):
            self.trajectory_manager.BuildTrajectories(self.actor)
            