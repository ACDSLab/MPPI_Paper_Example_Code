import abc
import math
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
from typing import Optional, List, Dict

Settings = namedtuple("Settings", """dt length_x length_y resolution
        origin_x origin_y costmap_default_val footprint_size
        obstacles obstacle_cost iteration_count num_timesteps
        lookahead_dist Lambda std_dev_v std_dev_omega v_max
        v_min omega_max goal_weight goal_power goal_dist_threshold, L, r""")

Obstacle = namedtuple("Obstacle", "x y size")
obstacles = [Obstacle(8, 8, 0.4), Obstacle(4, 6, 0.5), Obstacle(6, 5, 0.3)]
common_settings = Settings(0.02, 11, 11, 0.1, 0, 0, 0, 0.15, obstacles, 250, 1, 257, 10, 1.0, 0.2, 0.2, 0.5, -0.35, 0.5,
                           5.0, 1, 1000, 1.0, 1.0)

@torch.jit.script
def dynamics(x: torch.Tensor, u: torch.Tensor, settings: Dict[str, float], device: torch.device) -> torch.Tensor:
    #TODO Implement differential dynamics
    u = torch.clamp(u, min=torch.tensor([settings["v_min"], -settings["omega_max"]], device=device),
                       max=torch.tensor([settings["v_max"], settings["omega_max"]], device=device))
    next_state = x
    next_state[..., 0] += settings["r"] / 2 * (u[..., 0] + u[..., 1]) * torch.cos(x[..., 2]) * settings["dt"]
    next_state[..., 1] += settings["r"] / 2 * (u[..., 0] + u[..., 1]) * torch.sin(x[..., 2]) * settings["dt"]
    next_state[..., 2] += settings["r"] / settings["L"] * (u[..., 0] - u[..., 1]) * settings["dt"]
    return next_state

@torch.jit.script
def comparisionCost(x: torch.Tensor, u: torch.Tensor, obstacle_map: torch.Tensor, settings: Dict[str, float], device: torch.device) -> torch.Tensor:
    #TODO Implement cost function located in include/mppi_paper_example/costs/ComparisonCost/comparison_cost.cu
    x_diff = x[..., 0] - settings["goal_x"]
    y_diff = x[..., 1] - settings["goal_y"]
    dist = torch.hypot(x_diff, y_diff)

    yaw_diff = (x[..., 2] - settings["goal_yaw"] + math.pi) % 2 * math.pi - math.pi
    # query map
    normalized_x = x[..., 0] / settings["resolution"] - 0.5
    normalized_y = x[..., 1] / settings["resolution"] - 0.5
    normalized_x = torch.where(normalized_x > settings["rows"] - 1, settings["rows"] - 1, normalized_x)
    normalized_x = torch.where(normalized_x < 0, 0, normalized_x)
    normalized_y = torch.where(normalized_y > settings["cols"] - 1, settings["cols"] - 1, normalized_y)
    normalized_y = torch.where(normalized_y < 0, 0, normalized_y)
    max_x_val = torch.tensor(settings["rows"] - 2, device=device)
    max_y_val = torch.tensor(settings["cols"] - 2, device=device)
    x_min = torch.minimum(normalized_x, max_x_val).to(torch.int)
    y_min = torch.minimum(normalized_y, max_y_val).to(torch.int)
    x_max = x_min + 1
    y_max = y_min + 1
    q_11 = torch.index_select(obstacle_map.flatten(), 0, x_min + int(settings["cols"]) * y_min)
    q_12 = torch.index_select(obstacle_map.flatten(), 0, x_max + int(settings["cols"]) * y_min)
    q_21 = torch.index_select(obstacle_map.flatten(), 0, x_min + int(settings["cols"]) * y_max)
    q_22 = torch.index_select(obstacle_map.flatten(), 0, x_max + int(settings["cols"]) * y_max)
    y_min_interp = q_11 * ((x_max - normalized_x) ) * q_12 * ((normalized_x - x_min) )
    y_max_interp = q_21 * ((x_max - normalized_x) ) * q_22 * ((normalized_x - x_min) )
    obstacle_map_cost = y_min_interp * ((y_max - normalized_y) ) + y_max_interp * ((normalized_y - y_min) )

    distance_obstacle = (settings["obs_scale_factor"] * settings["min_radius"] - torch.log(obstacle_map_cost) + math.log(253.0)) / settings["obs_scale_factor"]
    obstacle_cost = torch.where(distance_obstacle < settings["collision_margin_dist"], settings["obs_traj_weight"] * (settings["collision_margin_dist"] - distance_obstacle), settings["obs_repulsion_weight"] * (settings["inflation_radius"] - distance_obstacle))
    obstacle_cost = torch.pow(obstacle_cost, settings["obs_power"])
    obstacle_cost = torch.where(obstacle_map_cost > settings["lethal_obstacle"], settings["collision_cost"], obstacle_cost)
    obstacle_cost = torch.where(obstacle_map_cost < 1, 0, obstacle_cost)

    # Combine costs
    goal_cost = torch.pow(settings["goal_weight"] * dist, settings["goal_power"]) + torch.pow(settings["angle_weight"] * torch.abs(yaw_diff), settings["angle_power"])
    cost = torch.where(dist < settings["goal_dist_threshold"], goal_cost, 0) + obstacle_cost

    # negate as we are calculating a cost, not a reward
    return -cost
    # tensordict.set("reward", -cost)

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
        self.dyn_dict = {"r": self.r, "L": self.L, "dt": self.dt,
                         "v_min": settings.v_min,
                         "v_max": settings.v_max,
                         "omega_max": settings.omega_max,}
        self.cost_dict = {"obs_scale_factor": self.obs_scale_factor,
                          "min_radius": self.min_radius,
                          "lethal_obstacle": self.lethal_obstacle,
                          "collision_cost": self.collision_cost,
                          "collision_margin_dist": self.collision_margin_dist,
                          "obs_traj_weight": self.obs_traj_weight,
                          "obs_repulsion_weight": self.obs_repulsion_weight,
                          "inflation_radius": self.inflation_radius,
                          "goal_weight": self.goal_weight,
                          "goal_power": self.goal_power,
                          "goal_yaw": self.goal_yaw,
                          "goal_dist_threshold": self.goal_dist_threshold,
                          "angle_weight": self.angle_weight,
                          "angle_power": self.angle_power,
                          "goal_x": self.goal[0],
                          "goal_y": self.goal[1],
                          "rows": self.rows,
                          "cols": self.cols,
                          "resolution": self.resolution,
                          "obs_power": self.obs_power,
                          }
        obstacle_map = np.zeros(shape=(self.rows, self.cols))
        for obstacle in settings.obstacles:
            obs_x = int(obstacle.x / settings.resolution)
            obs_y = int(obstacle.y / settings.resolution)
            obs_size = int(obstacle.size / settings.resolution)
            for row in range(obs_x, obs_x + obs_size):
                for col in range(obs_y, obs_y + obs_size):
                    obstacle_map[row, col] = settings.obstacle_cost

        self.map = torch.tensor(obstacle_map, device=self.device)

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

        # # run dynamics and cost on given state and action
        curr_x = tensordict.get("hidden_observation")
        curr_u = tensordict.get("action")
        next_x = dynamics(curr_x, curr_u, self.dyn_dict, self.device)
        reward = comparisionCost(next_x, curr_u, self.map, self.cost_dict, self.device)

        # set outputs
        tensordict.set("hidden_observation", next_x)
        tensordict.set("reward", reward)
        return tensordict.select(
                *self.observation_spec.keys(),
                *self.full_reward_spec.keys(),
                strict=False)


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
    print(torch.cuda.get_device_name())
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
        world_env.rollout(1, planner) # run outside of timing as the first run is slower than the following
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
