#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <mppi/dynamics/cartpole/cartpole_dynamics.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi_paper_example/plants/cartpole_plant.hpp>

const int NUM_TIMESTEPS = 100;
const int NUM_ROLLOUTS = 2048;
const int DYN_BLOCK_X = 64;
using DYN_T = CartpoleDynamics;
const int DYN_BLOCK_Y = DYN_T::STATE_DIM;
using COST_T = CartpoleQuadraticCost;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;
using CONTROLLER_T = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;
using CONTROLLER_PARAMS_T = CONTROLLER_T::TEMPLATED_PARAMS;

using PLANT_T = SimpleCartpolePlant<CONTROLLER_T>;

int main(int argc, char** argv)
{
  float dt = 0.02;
  DYN_T dynamics;                     // set up dynamics
  COST_T cost;                        // set up cost
  FB_T fb_controller(&dynamics, dt);  // set up feedback controller
  // set up sampling distribution
  SAMPLING_T sampler;
  auto sampler_params = sampler.getParams();
  std::fill(sampler_params.std_dev, sampler_params.std_dev + DYN_T::CONTROL_DIM, 1.0);
  sampler.setParams(sampler_params);

  // set up MPPI Controller
  CONTROLLER_PARAMS_T controller_params;
  controller_params.dt_ = dt;
  controller_params.lambda_ = 1.0;
  controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1);
  controller_params.cost_rollout_dim_ = dim3(96, 1, 1);
  std::shared_ptr<CONTROLLER_T> controller =
      std::make_shared<CONTROLLER_T>(&dynamics, &cost, &fb_controller, &sampler, controller_params);

  // Create plant
  PLANT_T plant(controller, (1.0 / dt), 1);

  std::atomic<bool> alive(true);
  for (int t = 0; t < 10000; t++)
  {
    plant.updateState(plant.current_state_, t * dt);
    plant.runControlIteration(&alive);
  }

  std::cout << "Avg Optimization time: " << plant.getAvgOptimizationTime() << " ms" << std::endl;
  std::cout << "Last Optimization time: " << plant.getLastOptimizationTime() << " ms" << std::endl;
  std::cout << "Avg Loop time: " << plant.getAvgLoopTime() << " ms" << std::endl;
  std::cout << "Avg Optimization Hz: " << 1.0 / (plant.getAvgOptimizationTime() * 1e-3) << " Hz" << std::endl;

  auto control_sequence = controller->getControlSeq();
  std::cout << "State: \n" << plant.current_state_.transpose() << std::endl;
  std::cout << "Control Sequence:\n" << control_sequence << std::endl;
  return 0;
}
