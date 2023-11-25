import abc
import numpy as np

# installed using `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
import torch
import torch.nn as nn

from collections import namedtuple
from tensordict import TensorDict

# installed with `pip3 install torchrl`
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.modules import MPPIPlanner, ValueOperator
from torchrl.objectives.value import TDLambdaEstimator
from torchrl.envs.common import EnvBase
from typing import Optional, List

Settings = namedtuple("Settings", """dt length_x length_y resolution
        origin_x origin_y costmap_default_val footprint_size
        obstacles obstacle_cost iteration_count num_timesteps
        lookahead_dist Lambda std_dev_v std_dev_omega v_max
        v_min omega_max goal_weight goal_power goal_dist_threshold""")

Obstacle = namedtuple("Obstacle", "x y size")
obstacles = [Obstacle(8, 8, 0.4), Obstacle(4, 6, 0.5), Obstacle(6, 5, 0.3)]
common_settings = Settings(0.02, 11, 11, 0.1, 0, 0, 0, 0.15, obstacles, 250, 1, 257, 10, 1.0, 0.2, 0.2, 0.5, -0.35, 0.5,
                           5.0, 1, 1000)

class DifferentialDriveEnv(EnvBase):
    # TODO: figure out how to get batch size [32] to work
    def __init__(self, settings: Settings, device="cuda", dtype=None, batch_size: torch.Size = None):
        super(DifferentialDriveEnv, self).__init__(device=device, dtype=dtype, batch_size=batch_size)
        self.dt = settings.dt

        self.state_spec = CompositeSpec(
                hidden_observation=UnboundedContinuousTensorSpec((4,))
        )
        self.observation_spec = CompositeSpec(
                hidden_observation=UnboundedContinuousTensorSpec((4,))
        )
        self.action_spec = CompositeSpec(
                hidden_observation=UnboundedContinuousTensorSpec((2,))
        )
        self.reward_spec = CompositeSpec(
                hidden_observation=UnboundedContinuousTensorSpec((1,))
        )

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(
            cls, *args, _inplace_update=False, _batch_locked=False, **kwargs
        )

    def _step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        # step method requires to be immutable
        print(tensordict)
        tensordict_out = tensordict.clone(recurse=False)
        # # run dynamics and cost on given state and action
        self.dynamics(tensordict_out)
        # cost(tensordict_out)
        return tensordict_out.select(
                *self.observation_spec.keys(),
                *self.full_reward_spec.keys(),
                strict=False)

    # @abc.abstractmethod
    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        self.curr_state = self.full_state_spec.zero()
        tensordict = TensorDict(
                {},
                batch_size=self.batch_size,
                device=self.device,
            )
        tensordict = tensordict.update(self.curr_state)
        tensordict = tensordict.update(self.full_action_spec.rand())
        tensordict = tensordict.update(self.curr_state)
        return tensordict

    def _set_seed(self, seed: Optional[int]) -> int:
        print("Dynamics and Cost function do not use seed")
        return seed

    def dynamics(self, tensordict: TensorDict):
        u = tensordict.get("action")
        curr_state = tensordict.get("hidden_observation")
        #TODO Implement differential dynamics
        print (f"State:\n{curr_state},\nU:\n{u}")
        self.curr_state[0] = self.curr_state[0] + self.dt

    def cost(self, tensordict: TensorDict):
        print (f"Input: {tensordict}")
        #TODO Implement cost function located in include/mppi_paper_example/costs/ComparisonCost/comparison_cost.cu

if __name__ == "__main__":
    world_env = DifferentialDriveEnv(settings=common_settings)
    value_net = nn.Linear(4, 1)
    value_net = ValueOperator(value_net, in_keys=["hidden_observation"])
    adv = TDLambdaEstimator(
        gamma=0.99,
        lmbda=0.95,
        value_network=value_net,
    )
    # Build a planner and use it as actor
    planner = MPPIPlanner(
        world_env,
        adv,
        temperature=1.0,
        planning_horizon=10,
        optim_steps=1,
        num_candidates=7,
        top_k=3)
    world_env.rollout(5, planner)
