#include <gtest/gtest.h>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi_paper_example/controllers/DMD-MPPI/dmd_mppi_controller.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
// #include <mppi_paper_example/costs/ComparisonCost/comparison_cost.cuh>
// #include <mppi_paper_example/dynamics/diff_drive/diff_drive.cuh>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi_paper_example/plants/sim_plant/sim_plant.hpp>

#include "test/common.hpp"

#include <cnpy.h>

#define USE_NEW_API

const int NUM_TIMESTEPS = 100;
using DYN_T = DoubleIntegratorDynamics;
using COST_T = DoubleIntegratorCircleCost;
// using COST_T = QuadraticCost<DYN_T>;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;

template <int NUM_ROLLOUTS = 128>
using MPPI_CONTROLLER_TEMPLATE = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;

template <int NUM_ROLLOUTS = 128>
using DMD_MPPI_CONTROLLER_TEMPLATE = DMDMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;

class DITestEnvironment : public ::testing::Environment
{
public:
  static std::ofstream csv_file;
  static std::string cpu_name;
  static std::string gpu_name;
  static std::string timestamp;
  ~DITestEnvironment() override {}
  void SetUp() override {
    timestamp = getTimestamp();
    std::string custom_csv_header =
        "Processor,GPU,Method,Step Size,Num Rollouts,Cost Min,Cost Mean,Cost Variance,Mean Optimization Time (ms), "
        "Std. Dev. Time (ms)\n";
    createNewCSVFile("mppi_dmd_comparisons", csv_file, custom_csv_header);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    gpu_name = std::string(deviceProp.name);

    // get CPU name
    cpu_name = getCPUModelName();
  }

  void TearDown() override{
    csv_file.close();
  }

  static std::string getGPUName()
  {
    return gpu_name;
  }

  static std::string getCPUName()
  {
    return cpu_name;
  }
};

// Iniitialize static variables
std::ofstream DITestEnvironment::csv_file;
std::string DITestEnvironment::cpu_name = "N/A";
std::string DITestEnvironment::gpu_name = "N/A";
std::string DITestEnvironment::timestamp = "0000";

// Register Environment
testing::Environment* const csv_env = testing::AddGlobalTestEnvironment(new DITestEnvironment);

template <class CONTROLLER_T>
class DMDMPPITest : public ::testing::Test
{
public:
  using CONTROLLER_PARAMS_T = typename CONTROLLER_T::TEMPLATED_PARAMS;
  using PLANT_T = SimpleDynPlant<CONTROLLER_T>;

protected:
  float dt = 0.02f;
  float std_dev = 2.0f;
  // DYN_T::state_array origin_eigen;
  // DYN_T::state_array goal;
  // DYN_T::state_array q_coeffs;
  // float origin[DYN_T::STATE_DIM] = { -9, -9, 0.1, 0.1 };
  float origin[DYN_T::STATE_DIM] = { -2, 0.0, 0.0, 0.0 };
  float goal[DYN_T::STATE_DIM] = { -4, -4, 0, 0 };
  float q_coeffs[DYN_T::STATE_DIM] = { 1.0, 1.0, 0, 0 };
  float lambda = 1.0f;
  float alpha = 0.0f;
  int max_iter = 1;
  unsigned int simulation_time_horizon = 1000;

  const int DYN_BLOCK_X = 64;
  const int DYN_BLOCK_Y = DYN_T::STATE_DIM;
  const int COST_BLOCK_X = 64;

  std::shared_ptr<CONTROLLER_T> controller;
  DYN_T* dynamics = nullptr;
  COST_T* cost = nullptr;
  FB_T* fb_controller = nullptr;
  SAMPLING_T* sampler = nullptr;
  PLANT_T* plant = nullptr;
  CONTROLLER_PARAMS_T controller_params;
  cudaStream_t stream;

  void SetUp() override
  {
    SAMPLING_T::SAMPLING_PARAMS_T sampler_params;
    sampler_params.std_dev[C_IND_CLASS(DYN_T::DYN_PARAMS_T, ACCEL_X)] = std_dev;
    sampler_params.std_dev[C_IND_CLASS(DYN_T::DYN_PARAMS_T, ACCEL_Y)] = std_dev;
    sampler = new SAMPLING_T(sampler_params);

    /**
     * Set up dynamics
     **/
    dynamics = new DYN_T();

    /**
     * Set up Cost function
     **/
    cost = new COST_T();
    // auto cost_params = cost->getParams();
    // for (int i = 0; i < DYN_T::STATE_DIM; i++)
    // {
    //   cost_params.s_coeffs[i] = q_coeffs[i];
    //   cost_params.s_goal[i] = goal[i];
    // }
    // cost->setParams(cost_params);

    /**
     * Set up Feedback Controller
     **/
    fb_controller = new FB_T(dynamics, dt);

    /**
     * Set up Controller
     **/
    controller_params.dt_ = dt;
    controller_params.lambda_ = lambda;
    controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1);
    controller_params.cost_rollout_dim_ = dim3(COST_BLOCK_X, 1, 1);
    controller_params.num_iters_ = max_iter;
    controller_params.seed_ = 42;

    HANDLE_ERROR(cudaStreamCreate(&stream));

    controller = std::make_shared<CONTROLLER_T>(dynamics, cost, fb_controller, sampler, controller_params, stream);
    controller->setLogLevel(mppi::util::LOG_LEVEL::DEBUG);
    controller->chooseAppropriateKernel();

    plant = new PLANT_T(controller, 1.0f / dt, 1);
    for (int i = 0; i < DYN_T::STATE_DIM; i++)
    {
      this->plant->current_state_(i) = origin[i];
    }
  }

  void TearDown() override
  {
    delete plant;
    controller.reset();
    delete fb_controller;
    delete cost;
    delete dynamics;
    delete sampler;
  }
};

using DIFFERENT_CONTROLLERS = ::testing::Types<DMD_MPPI_CONTROLLER_TEMPLATE<64>, DMD_MPPI_CONTROLLER_TEMPLATE<128>,
                                               DMD_MPPI_CONTROLLER_TEMPLATE<256>, DMD_MPPI_CONTROLLER_TEMPLATE<512>,
                                               DMD_MPPI_CONTROLLER_TEMPLATE<1024>, DMD_MPPI_CONTROLLER_TEMPLATE<2048> >;

TYPED_TEST_SUITE(DMDMPPITest, DIFFERENT_CONTROLLERS);

TYPED_TEST(DMDMPPITest, DifferentNumSamples)
{
  const int num_iterations = 1000;
  const int num_steps = 10;
  const float min_step_size = 0.5f;
  const float step_size_range = 0.5f;
  RunningStats<double> times, cost_stats;
  std::atomic<bool> alive(true);
  std::vector<float> states_taken(DYN_T::STATE_DIM * this->simulation_time_horizon);
  std::vector<float> best_states(DYN_T::STATE_DIM * this->simulation_time_horizon);
  std::vector<float> controls_taken(DYN_T::CONTROL_DIM * this->simulation_time_horizon);
  std::vector<float> best_controls(DYN_T::CONTROL_DIM * this->simulation_time_horizon);
  std::vector<float> costs_taken(this->simulation_time_horizon), best_costs(this->simulation_time_horizon);
  float min_cost = std::numeric_limits<float>::infinity();
  float total_cost = 0.0f;
  int crash_status = 0;
  DYN_T::state_array initial_state = this->plant->current_state_;
  for (int s = 0; s <= num_steps; s++)
  {
    this->controller_params.step_size = (float)s / num_steps * step_size_range + min_step_size;
    this->controller->setParams(this->controller_params);
    times.reset();
    cost_stats.reset();
    for (int it = 0; it < num_iterations; it++)
    {
      total_cost = 0.0f;
      crash_status = 0;
      this->plant->current_state_ = initial_state;
      this->plant->resetStateTime();
      this->controller->updateImportanceSampler(this->controller_params.init_control_traj_);
      states_taken.reserve(DYN_T::STATE_DIM * this->simulation_time_horizon);
      controls_taken.reserve(DYN_T::CONTROL_DIM * this->simulation_time_horizon);
      costs_taken.reserve(this->simulation_time_horizon);
      for (int t = 0; t < this->simulation_time_horizon; t++)
      {
        auto start = std::chrono::steady_clock::now();
        this->plant->updateState(this->plant->current_state_, (t + 1) * this->dt);
        this->plant->runControlIteration(&alive);
        auto end = std::chrono::steady_clock::now();
        double duration = (end - start).count() / 1e6;
        times.add(duration);

        // save out control and state
        for (int i = 0; i < DYN_T::STATE_DIM; i++)
        {
          states_taken[t * DYN_T::STATE_DIM + i] = this->plant->current_state_.data()[i];
        }
        for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
        {
          controls_taken[t * DYN_T::CONTROL_DIM + i] = this->controller->getControlSeq().col(1)(i, 0);
        }
        total_cost += this->cost->computeRunningCost(this->plant->current_state_,
                                                     this->controller->getControlSeq().col(1), t, &crash_status);
        costs_taken[t] = total_cost;
      }
      cost_stats.add(total_cost);
      if ((it + 1) % 50 == 0)
      {
        printf("Finished iteration %5d/%5d with cost %f\r", it + 1, num_iterations, total_cost);
        fflush(stdout);
      }

      // Keep the best trajectory
      if (total_cost < min_cost)
      {
        best_costs = std::move(costs_taken);
        best_states = std::move(states_taken);
        best_controls = std::move(controls_taken);
        min_cost = total_cost;
      }
    }
    std::string step_size_string = std::to_string(this->controller_params.step_size).substr(0, 4);
    std::string cnpy_file_name = this->controller->getControllerName() + "_" +
                                 std::to_string(this->controller->sampler_->getNumRollouts()) + "_" + step_size_string +
                                 "_" + DITestEnvironment::timestamp + ".npy";
    cnpy::npy_save("state_" + cnpy_file_name, best_states.data(), { this->simulation_time_horizon, DYN_T::STATE_DIM },
                   "w");
    cnpy::npy_save("control_" + cnpy_file_name, best_controls.data(),
                   { this->simulation_time_horizon, DYN_T::CONTROL_DIM }, "w");
    cnpy::npy_save("costs_" + cnpy_file_name, best_costs.data(), { this->simulation_time_horizon, 1 }, "w");
    // Save to CSV File
    DITestEnvironment::csv_file << DITestEnvironment::getCPUName() << "," << DITestEnvironment::getGPUName() << ","
                                << this->controller->getControllerName() << "," << step_size_string << ","
                                << this->controller->sampler_->getNumRollouts() << "," << cost_stats.min() << ","
                                << cost_stats.mean() << "," << cost_stats.variance() << "," << times.mean() << ","
                                << sqrt(times.variance()) << std::endl;
    printf("MPPI-Generic %s with %d rollouts optimization time (%s step size): %f +- %f ms and cost %f +- %f\n",
           this->controller->getControllerName().c_str(), this->controller->sampler_->getNumRollouts(),
           step_size_string.c_str(), times.mean(), sqrt(times.variance()), cost_stats.mean(),
           sqrt(cost_stats.variance()));
    printf("\tAverage Optimization Hz: %f Hz\n", 1000.0 / times.mean());
  }
}
