from PPO.TrajectoryManager import TrajectoryManager
from PPO.Driver import PPODriver
import numpy as np

driver=PPODriver(actions=[
                0.2,
                0.5,
                1.0,
                2.0,
                1000.0
            ]
)

driver.run()

