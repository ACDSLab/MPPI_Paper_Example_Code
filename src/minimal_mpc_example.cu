#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/dynamics/cartpole/cartpole_dynamics.cuh>
#include <mppi/cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi_paper_example/cartpole_plant.hpp>

#include <stdio.h>

#define USE_NEW_API

const int NUM_TIMESTEPS = 100; const int NUM_ROLLOUTS = 8192 * 4;
const int DYN_BLOCK_X = 256;
using DYN_T = CartpoleDynamics;
const int DYN_BLOCK_Y = DYN_T::STATE_DIM;
// const int DYN_BLOCK_Y = 1;
using COST_T = CartpoleQuadraticCost;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
#ifdef USE_NEW_API
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;
using CONTROLLER_T = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;
#else
using CONTROLLER_T = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, DYN_BLOCK_X, DYN_BLOCK_Y>;
#endif
using CONTROLLER_PARAMS_T = CONTROLLER_T::TEMPLATED_PARAMS;

using PLANT_T = SimpleCartpolePlant<CONTROLLER_T>;

int main (int argc, char** argv){
  float dt = 0.02;
  // set up dynamics
  float cart_mass = 1.0;
  float pole_mass = 1.0;
  float pole_length = 1.0;
  DYN_T dynamics(cart_mass, pole_mass, pole_length);
  // set up cost
  COST_T cost;
  // set up feedback controller
  FB_T fb_controller(&dynamics, dt);
#ifdef USE_NEW_API
  SAMPLING_T sampler;
  auto sampler_params = sampler.getParams();
  for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 1.0;
  }
  sampler.setParams(sampler_params);
#endif

  // set up MPPI Controller
  CONTROLLER_PARAMS_T controller_params;
  controller_params.dt_ = dt;
  controller_params.lambda_ = 1.0;
#ifndef USE_NEW_API
  controller_params.control_std_dev_ = DYN_T::control_array::Ones();
  std::shared_ptr<CONTROLLER_T> controller = std::make_shared<CONTROLLER_T>(&dynamics, &cost, &fb_controller, controller_params);
#else
  controller_params.dynamics_rollout_dim_= dim3(DYN_BLOCK_X, 4, 1);
  controller_params.cost_rollout_dim_ = dim3(96, 1, 1);
  std::shared_ptr<CONTROLLER_T> controller = std::make_shared<CONTROLLER_T>(&dynamics, &cost, &fb_controller, &sampler, controller_params);
#endif

  PLANT_T plant(controller, (1.0 / dt), 1);

  std::atomic<bool> alive(true);
  for (int t = 0; t < 10000; t++)
  {
    plant.updateState(plant.current_state_, t * dt);
    plant.runControlIteration(&alive);
    // std::cout << "t: " << t * dt << ", state: " << plant.current_state_.transpose() << std::endl;
  }

  std::cout << "Average Optimization time: " << plant.getAvgOptimizationTime() << " ms" << std::endl;
  std::cout << "Last Optimization time: " << plant.getLastOptimizationTime() << " ms" << std::endl;
  std::cout << "Avg Loop time: " << plant.getAvgLoopTime() << " ms" << std::endl;
  std::cout << "Average Optimization Hz: " << 1.0f / (plant.getAvgOptimizationTime() * 1e-3f) << " ms" << std::endl;

  auto control_sequence = controller->getControlSeq();
  std::cout << "State: \n" << plant.current_state_.transpose() << std::endl;
  std::cout << "Control Sequence:\n" << control_sequence << std::endl;
  return 0;
}
