import abc
import numpy as np
import time

# installed using `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
import torch
import torch.nn as nn

from collections import namedtuple
from tensordict import TensorDict
from tqdm import tqdm

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
        self.goal = np.array([8, 8])
        self.goal_weight = settings.goal_weight
        self.goal_power = settings.goal_power
        self.goal_dist_threshold = settings.goal_dist_threshold
        self.goal_yaw = 0
        self.angle_weight = 1
        self.angle_power = 1

        self.r = 1.0
        self.L = 1.0
        self.rows = int(settings.length_x / settings.resolution)
        self.cols = int(settings.length_y / settings.resolution)
        self.resolution = settings.resolution
        self.obs_scale_factor = 1.0
        self.min_radius = 0.1
        self.lethal_obstacle = 1e10
        self.collision_cost = 10000.0
        self.collision_margin_dist = 0.1
        self.obs_traj_weight = 20
        self.obs_repulsion_weight = 0.0
        self.inflation_radius = 0.1
        self.obs_power = 1
        obstacle_map = np.zeros(shape=(self.rows, self.cols))
        for obstacle in settings.obstacles:
            obs_x = int(obstacle.x / settings.resolution)
            obs_y = int(obstacle.y / settings.resolution)
            obs_size = int(obstacle.size / settings.resolution)
            for row in range(obs_x, obs_x + obs_size):
                for col in range(obs_y, obs_y + obs_size):
                    obstacle_map[row, col] = settings.obstacle_cost

        self.map = torch.tensor(obstacle_map, device=self.device)
        print(self.map)

        self.state_spec = CompositeSpec(
                hidden_observation=UnboundedContinuousTensorSpec((3,))
        )
        self.observation_spec = CompositeSpec(
                hidden_observation=UnboundedContinuousTensorSpec((3,))
        )
        self.action_spec = CompositeSpec(
                hidden_observation=UnboundedContinuousTensorSpec((2,))
        )
        self.reward_spec = CompositeSpec(
                reward=UnboundedContinuousTensorSpec((1,))
        )

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(
            cls, *args, _inplace_update=False, _batch_locked=False, **kwargs
        )

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

    def _step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        # step method requires to be immutable
        tensordict_out = tensordict.clone(recurse=False)

        # Ensure inputs are always [batch size, dim]
        if len(tensordict.get("hidden_observation").shape) == 1:
            for key in tensordict_out.keys():
                temp_vec = torch.unsqueeze(tensordict_out.get(key), dim=0)
                tensordict_out.set(key, temp_vec)
        # # run dynamics and cost on given state and action
        self.dynamics(tensordict_out)
        self.cost(tensordict_out)

        # set back to original dimensions
        if len(tensordict.get("hidden_observation").shape) == 1:
            for key in tensordict_out.keys():
                temp_vec = torch.squeeze(tensordict_out.get(key), dim=0)
                tensordict_out.set(key, temp_vec)
        return tensordict_out.select(
                *self.observation_spec.keys(),
                *self.full_reward_spec.keys(),
                strict=False)

    def dynamics(self, tensordict: TensorDict):
        u = tensordict.get("action")
        curr_state = tensordict.get("hidden_observation")
        #TODO Implement differential dynamics
        next_state = curr_state
        next_state[:, 0] += self.r / 2 * (u[:, 0] + u[:, 1]) * torch.cos(curr_state[:, 2]) * self.dt
        next_state[:, 1] += self.r / 2 * (u[:, 0] + u[:, 1]) * torch.sin(curr_state[:, 2]) * self.dt
        next_state[:, 2] += self.r / self.L * (u[:, 0] - u[:, 1]) * self.dt

        tensordict.set("hidden_observation", next_state)

    def cost(self, tensordict: TensorDict):
        #TODO Implement cost function located in include/mppi_paper_example/costs/ComparisonCost/comparison_cost.cu
        curr_x = tensordict.get("hidden_observation")
        curr_u = tensordict.get("action")
        x_diff = curr_x[:, 0] - self.goal[0]
        y_diff = curr_x[:, 1] - self.goal[1]
        dist = torch.hypot(x_diff, y_diff)

        yaw_diff = (curr_x[:, 2] - self.goal_yaw + np.pi) % 2 * np.pi - np.pi
        # query map
        normalized_x = curr_x[:, 0] / self.resolution - 0.5
        normalized_y = curr_x[:, 1] / self.resolution - 0.5
        normalized_x = torch.where(normalized_x > self.rows - 1, self.rows - 1, normalized_x)
        normalized_x = torch.where(normalized_x < 0, 0, normalized_x)
        normalized_y = torch.where(normalized_y > self.cols - 1, self.cols - 1, normalized_y)
        normalized_y = torch.where(normalized_y < 0, 0, normalized_y)
        max_x_val = torch.tensor(self.rows - 2, device=self.device)
        max_y_val = torch.tensor(self.cols - 2, device=self.device)
        x_min = torch.minimum(normalized_x, max_x_val).to(torch.int)
        y_min = torch.minimum(normalized_y, max_y_val).to(torch.int)
        x_max = x_min + 1
        y_max = y_min + 1
        q_11 = torch.tensor([self.map[x, y] for x, y in zip(x_min, y_min)], device=self.device)
        q_12 = torch.tensor([self.map[x, y] for x, y in zip(x_max, y_min)], device=self.device)
        q_21 = torch.tensor([self.map[x, y] for x, y in zip(x_min, y_max)], device=self.device)
        q_22 = torch.tensor([self.map[x, y] for x, y in zip(x_max, y_max)], device=self.device)
        y_min_interp = q_11 * ((x_max - normalized_x) / (x_max - x_min)) * q_12 * ((normalized_x - x_min) / (x_max - x_min))
        y_max_interp = q_21 * ((x_max - normalized_x) / (x_max - x_min)) * q_22 * ((normalized_x - x_min) / (x_max - x_min))
        obstacle_map_cost = y_min_interp * ((y_max - normalized_y) / (y_max - y_min)) + y_max_interp * ((normalized_y - y_min) / (y_max - y_min))

        distance_obstacle = (self.obs_scale_factor * self.min_radius - torch.log(obstacle_map_cost) + np.log(253.0)) / self.obs_scale_factor
        obstacle_cost = torch.where(distance_obstacle < self.collision_margin_dist, self.obs_traj_weight * (self.collision_margin_dist - distance_obstacle), self.obs_repulsion_weight * (self.inflation_radius - distance_obstacle))
        obstacle_cost = torch.pow(obstacle_cost, self.obs_power)
        obstacle_cost = torch.where(obstacle_map_cost > self.lethal_obstacle, self.collision_cost, distance_obstacle)
        obstacle_cost = torch.where(obstacle_map_cost < 1, 0, obstacle_cost)

        # Combine costs
        goal_cost = torch.pow(self.goal_weight * dist, self.goal_power) + torch.pow(self.angle_weight * torch.abs(yaw_diff), self.angle_power)
        reward = torch.where(dist < self.goal_dist_threshold, goal_cost, 0) + obstacle_cost

        # negate as we are calculating a cost, not a reward
        tensordict.set("reward", -reward)

class BaselineExtractor (nn.Module):
    def __init__(self, in_key = "reward"):
        super().__init__()
        self.in_key = in_key

    def forward(self, x):
        output = x.get("next").get(self.in_key)
        baseline = output - torch.min(output)
        x.set("advantage", baseline)
        return x


class RunningStats:
    def __init__(self):
        self.clear()

    def add (self, val: float):
        self.count += 1
        if self.count == 1:
            self.mean_ = val
        else:
            new_mean = self.mean_ + (val - self.mean_) / self.count
            new_variance = self.variance_ + (val - self.mean_) * (val - new_mean)
            self.mean_ = new_mean
            self.variance_ = new_variance

    def mean(self):
        return self.mean_ if self.count > 0 else 0.0

    def variance(self):
        return self.variance_ / (self.count - 1) if self.count > 1 else 0.0

    def clear(self):
        self.mean_ = 0
        self.variance_ = 0
        self.count = 0

@torch.no_grad()
def main():
    world_env = DifferentialDriveEnv(settings=common_settings)
    value_net = nn.Linear(1, 1)
    value_net = ValueOperator(value_net, in_keys=["reward"])

    adv = BaselineExtractor(in_key = "reward")
    running_stats = RunningStats()
    num_iterations = 1000
    # Build a planner and use it as actor
    num_rollouts = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384]
    num_rollouts.reverse()
    for rollout_i in num_rollouts:
        planner = MPPIPlanner(
            world_env,
            adv,
            temperature=1.0,
            planning_horizon=100,
            optim_steps=1,
            num_candidates=rollout_i,
            top_k=rollout_i)
        running_stats.clear()
        for i in tqdm(range(num_iterations)):
            start = time.time()
            world_env.rollout(1, planner)
            end = time.time()
            running_stats.add(end - start)
        print("Torchrl MPPI with {} rollouts optimization time: {} +- {} ms".format(
                rollout_i, running_stats.mean() * 1000, np.sqrt(running_stats.variance()) * 1000
            ))
        print("\tAverage Optimization Hz: {} Hz".format(1.0 / running_stats.mean()))

if __name__ == "__main__":
    main()
