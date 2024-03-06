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
    createNewCSVFile("mppi_dmd_comparisons", csv_file);
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

using DIFFERENT_CONTROLLERS =
    ::testing::Types<MPPI_CONTROLLER_TEMPLATE<64>, DMD_MPPI_CONTROLLER_TEMPLATE<64>,
                     MPPI_CONTROLLER_TEMPLATE<128>, DMD_MPPI_CONTROLLER_TEMPLATE<128>,
                     // MPPI_CONTROLLER_TEMPLATE<256>, DMD_MPPI_CONTROLLER_TEMPLATE<256>,
                     // MPPI_CONTROLLER_TEMPLATE<512>, DMD_MPPI_CONTROLLER_TEMPLATE<512>,
                     MPPI_CONTROLLER_TEMPLATE<1024>, DMD_MPPI_CONTROLLER_TEMPLATE<1024>
                     >; //CONTROLLER_TEMPLATE<256>, CONTROLLER_TEMPLATE<512>,
                     // CONTROLLER_TEMPLATE<1024>, CONTROLLER_TEMPLATE<2048>, CONTROLLER_TEMPLATE<4096>,
                     // CONTROLLER_TEMPLATE<6144>, CONTROLLER_TEMPLATE<8192>, CONTROLLER_TEMPLATE<16384>>;

TYPED_TEST_SUITE(DMDMPPITest, DIFFERENT_CONTROLLERS);

TYPED_TEST(DMDMPPITest, DifferentNumSamples)
{
  RunningStats<double> times;
  std::atomic<bool> alive(true);
  std::vector<float> states_taken;
  std::vector<float> controls_taken;
  std::vector<float> costs_taken;
  states_taken.reserve(DYN_T::STATE_DIM * this->simulation_time_horizon);
  controls_taken.reserve(DYN_T::CONTROL_DIM * this->simulation_time_horizon);
  costs_taken.reserve(this->simulation_time_horizon);
  float total_cost = 0.0f;
  int crash_status = 0;
  for (int t = 0; t < this->simulation_time_horizon; t++)
  {
    auto start = std::chrono::steady_clock::now();
    this->plant->updateState(this->plant->current_state_, t * this->dt);
    this->plant->runControlIteration(&alive);
    auto end = std::chrono::steady_clock::now();
    double duration = (end - start).count() / 1e6;
    times.add(duration);

    // save out control and state
    for (int i = 0; i < DYN_T::STATE_DIM; i++)
    {
      states_taken[t * DYN_T::STATE_DIM + i] = this->plant->current_state_(i);
    }
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
      controls_taken[t * DYN_T::CONTROL_DIM + i] = this->controller->getControlSeq().col(1)(i, 0);
    }
    total_cost += this->cost->computeRunningCost(this->plant->current_state_, this->controller->getControlSeq().col(1),
                                                 t, &crash_status);
    costs_taken[t] = total_cost;
  }
  std::string cnpy_file_name = this->controller->getControllerName() + "_" + std::to_string(this->controller->sampler_->getNumRollouts())
                              + "_" + DITestEnvironment::timestamp + ".npy";
  cnpy::npy_save("state_" + cnpy_file_name, states_taken.data(), { this->simulation_time_horizon, DYN_T::STATE_DIM }, "w");
  cnpy::npy_save("control_" + cnpy_file_name, controls_taken.data(), { this->simulation_time_horizon, DYN_T::CONTROL_DIM }, "w");
  cnpy::npy_save("costs_" + cnpy_file_name, costs_taken.data(), { this->simulation_time_horizon, 1 }, "w");
  // Save to CSV File
  DITestEnvironment::csv_file << DITestEnvironment::getCPUName()
      << "," << DITestEnvironment::getGPUName() << "," << this->controller->getControllerName() << ","
      << this->controller->sampler_->getNumRollouts() << "," << times.mean()
      << "," << sqrt(times.variance()) << std::endl;
  printf("MPPI-Generic %s with %d rollouts optimization time: %f +- %f ms and cost %f\n", this->controller->getControllerName().c_str(),
         this->controller->sampler_->getNumRollouts(), times.mean(), sqrt(times.variance()), total_cost);
  printf("\tAverage Optimization Hz: %f Hz\n", 1000.0 / times.mean());
}
