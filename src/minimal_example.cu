#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <mppi/dynamics/cartpole/cartpole_dynamics.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi_paper_example/controllers/CEM/cem_controller.cuh>

const int NUM_TIMESTEPS = 100;
const int NUM_ROLLOUTS = 2048;

using DYN_T = CartpoleDynamics;
using COST_T = CartpoleQuadraticCost;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;
using CONTROLLER_T = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;
// using CONTROLLER_T = CEMController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;
using CONTROLLER_PARAMS_T = CONTROLLER_T::TEMPLATED_PARAMS;

int main(int argc, char** argv) {
  float dt = 0.02;
  // set up dynamics
  std::shared_ptr<DYN_T> dynamics = std::make_shared<DYN_T>();
  // set up cost
  std::shared_ptr<COST_T> cost = std::make_shared<COST_T>();
  // set up feedback controller
  std::shared_ptr<FB_T> fb_controller = std::make_shared<FB_T>(dynamics.get(), dt);
  // set up sampling distribution
  SAMPLING_T::SAMPLING_PARAMS_T sampler_params;
  std::fill(sampler_params.std_dev, sampler_params.std_dev + DYN_T::CONTROL_DIM, 1.0);
  std::shared_ptr<SAMPLING_T> sampler = std::make_shared<SAMPLING_T>(sampler_params);

  // set up MPPI Controller
  CONTROLLER_PARAMS_T controller_params;
  controller_params.dt_ = dt;
  controller_params.lambda_ = 1.0;
  controller_params.dynamics_rollout_dim_ = dim3(64, DYN_T::STATE_DIM, 1);
  controller_params.cost_rollout_dim_ = dim3(NUM_TIMESTEPS, 1, 1);
  std::shared_ptr<CONTROLLER_T> controller = std::make_shared<CONTROLLER_T>(
      dynamics.get(), cost.get(), fb_controller.get(), sampler.get(), controller_params);

  // set up initial state
  DYN_T::state_array x = dynamics->getZeroState();

  // calculate control
  controller->computeControl(x, 1);
  auto control_sequence = controller->getControlSeq();
  std::cout << "Control Sequence:\n" << control_sequence << std::endl;
  return 0;
}
