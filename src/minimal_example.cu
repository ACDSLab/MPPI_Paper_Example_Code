#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/dynamics/cartpole/cartpole_dynamics.cuh>
#include <mppi/cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>

#include <stdio.h>

#define USE_NEW_API

const int NUM_TIMESTEPS = 100;
const int NUM_ROLLOUTS = 2048;
const int DYN_BLOCK_X = 32;
using DYN_T = CartpoleDynamics;
const int DYN_BLOCK_Y = DYN_T::STATE_DIM;
using COST_T = CartpoleQuadraticCost;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
#ifdef USE_NEW_API
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;
using CONTROLLER_T = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;
#else
using CONTROLLER_T = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, DYN_BLOCK_X, DYN_BLOCK_Y>;
#endif

using CONTROLLER_PARAMS_T = CONTROLLER_T::TEMPLATED_PARAMS;

int main (int argc, char** argv){
  float dt = 0.02;
  // set up dynamics
  float cart_mass = 1.0;
  float pole_mass = 1.0;
  float pole_length = 1.0;
  DYN_T* dynamics = new DYN_T(cart_mass, pole_mass, pole_length);
  // set up cost
  COST_T* cost = new COST_T();
  // set up feedback controller
  FB_T* fb_controller = new FB_T(dynamics, dt);
#ifdef USE_NEW_API
  SAMPLING_T* sampler = new SAMPLING_T();
  auto sampler_params = sampler->getParams();
  for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 1.0;
  }
  sampler->setParams(sampler_params);
#endif

  // set up MPPI Controller
  CONTROLLER_PARAMS_T controller_params;
  controller_params.dt_ = dt;
  controller_params.lambda_ = 1.0;
#ifndef USE_NEW_API
  controller_params.control_std_dev_ = DYN_T::control_array::Ones();
  CONTROLLER_T* controller = new CONTROLLER_T(dynamics, cost, fb_controller, controller_params);
#else
  controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1);
  CONTROLLER_T* controller = new CONTROLLER_T(dynamics, cost, fb_controller, sampler, controller_params);
#endif

  // set up initial state
  DYN_T::state_array x = DYN_T::state_array::Zero();

  // calculate control
  controller->computeControl(x, 1);
  auto control_sequence = controller->getControlSeq();
  std::cout << "Control Sequence:\n" << control_sequence << std::endl;
  delete controller;
  delete fb_controller;
  delete cost;
  delete dynamics;
#ifdef USE_NEW_API
  delete sampler;
#endif
  return 0;
}
