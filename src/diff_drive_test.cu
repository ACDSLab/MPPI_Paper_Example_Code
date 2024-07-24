#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>

#include <mppi_paper_example/costs/diff_drive_cost/diff_drive_cost.cuh>
#include <mppi_paper_example/dynamics/diff_drive/diff_drive.cuh>
// #include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>

#include <stdio.h>

const int NUM_TIMESTEPS = 100;
const int NUM_ROLLOUTS = 2048;
const int DYN_BLOCK_X = 32;
using DYN_T = DiffDrive;
const int DYN_BLOCK_Y = DYN_T::STATE_DIM;
// using COST_T = QuadraticCost<DYN_T>;
using COST_T = DiffDriveCost;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;
using CONTROLLER_T = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;

using CONTROLLER_PARAMS_T = CONTROLLER_T::TEMPLATED_PARAMS;
using control_array = CONTROLLER_T::control_array;

int main(int argc, char** argv)
{
  float dt = 0.02;
  // set up dynamics
  DYN_T* dynamics = new DYN_T();
  // set up cost
  COST_T* cost = new COST_T();
  // set up feedback controller
  FB_T* fb_controller = new FB_T(dynamics, dt);
  SAMPLING_T* sampler = new SAMPLING_T();
  auto sampler_params = sampler->getParams();
  for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 1.0;
  }
  sampler->setParams(sampler_params);

  // set up MPPI Controller
  CONTROLLER_PARAMS_T controller_params;
  controller_params.dt_ = dt;
  controller_params.lambda_ = 1.0;
  controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1);
  CONTROLLER_T* controller = new CONTROLLER_T(dynamics, cost, fb_controller, sampler, controller_params);

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
  delete sampler;
  return 0;
}
