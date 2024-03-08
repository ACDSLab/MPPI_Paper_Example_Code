#include <gtest/gtest.h>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/cost_functions/autorally/ar_standard_cost.cuh>
#include <mppi/dynamics/autorally/ar_nn_model.cuh>
#include <mppi_paper_example/plants/sim_plant/sim_plant.hpp>

#include "test/common.hpp"
#include "test/mppi_controller.cuh"

#include <autorally_system_testing.h>

const int NUM_TIMESTEPS = AutorallySettings::num_timesteps;
using DYN_T = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>;
using COST_T = ARStandardCost;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;

template <int NUM_ROLLOUTS = 128>
using CONTROLLER_TEMPLATE = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;

template <int NUM_ROLLOUTS = 128>
using AUTORALLY_MPPI_TEMPLATE = autorally_control::MPPIController<DYN_T, COST_T, NUM_ROLLOUTS, AutorallySettings::DYN_BLOCK_X, AutorallySettings::DYN_BLOCK_Y>;

class CSVWritingEnvironment : public ::testing::Environment
{
public:
  static std::ofstream csv_file;
  static std::string cpu_name;
  static std::string gpu_name;
  ~CSVWritingEnvironment() override {}
  void SetUp() override {
    createNewCSVFile("autorally_sys_results", csv_file);
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
std::ofstream CSVWritingEnvironment::csv_file;
std::string CSVWritingEnvironment::cpu_name = "N/A";
std::string CSVWritingEnvironment::gpu_name = "N/A";

// Register Environment
testing::Environment* const csv_env = testing::AddGlobalTestEnvironment(new CSVWritingEnvironment);

template <class CONTROLLER_T>
class MPPIGenericAutorallyTest : public ::testing::Test
{
public:
  using CONTROLLER_PARAMS_T = typename CONTROLLER_T::TEMPLATED_PARAMS;
  using PLANT_T = SimpleDynPlant<CONTROLLER_T>;

protected:
  AutorallySettings settings;
  // const int DYN_BLOCK_X = 64;
  // const int DYN_BLOCK_Y = DYN_T::STATE_DIM;
  // const int COST_BLOCK_X = 64;

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
    sampler_params.std_dev[C_IND_CLASS(DYN_T::DYN_PARAMS_T, STEERING)] = settings.std_steering;
    sampler_params.std_dev[C_IND_CLASS(DYN_T::DYN_PARAMS_T, THROTTLE)] = settings.std_throttle;
    // sampler_params.rewrite_controls_block_dim.y = 8;
    sampler = new SAMPLING_T(sampler_params);

    /**
     * Set up dynamics
     **/
    dynamics = new DYN_T();
    dynamics->loadParams(mppi_generic_testing::nn_file);
    std::array<float2, DYN_T::CONTROL_DIM> control_ranges;
    control_ranges[C_IND_CLASS(DYN_T::DYN_PARAMS_T, STEERING)] = settings.steering_range;
    control_ranges[C_IND_CLASS(DYN_T::DYN_PARAMS_T, THROTTLE)] = settings.throttle_range;
    dynamics->setControlRanges(control_ranges);

    /**
     * Set up Cost function
     **/
    cost = new COST_T();
    auto cost_params = cost->getParams();
    cost_params.speed_coeff = settings.speed_coeff;
    cost_params.track_coeff = settings.track_coeff;
    cost_params.slip_coeff = settings.slip_coeff;
    cost_params.crash_coeff = settings.crash_coeff;
    cost_params.boundary_threshold = settings.boundary_threshold;
    cost_params.desired_speed = settings.desired_speed;
    cost_params.max_slip_ang = settings.max_slip_angle;
    cost_params.track_slop = settings.track_slop;
    cost_params.discount = settings.discount;

    cost->setParams(cost_params);

    // Setup cost map
    auto logger = std::make_shared<mppi::util::MPPILogger>();
    cost->GPUSetup();
    cost->loadTrackData(mppi_generic_testing::track_file);

    /**
     * Set up Feedback Controller
     **/
    fb_controller = new FB_T(dynamics, settings.dt);

    /**
     * Set up Controller
     **/
    controller_params.dt_ = settings.dt;
    controller_params.lambda_ = settings.lambda;
    controller_params.dynamics_rollout_dim_ = dim3(settings.DYN_BLOCK_X, settings.DYN_BLOCK_Y, 1);
    controller_params.cost_rollout_dim_ = dim3(settings.COST_BLOCK_X, 1, 1);
    controller_params.num_iters_ = settings.num_iters;
    // controller_params.optimization_stride_ = settings.optimization_stride;

    HANDLE_ERROR(cudaStreamCreate(&stream));

    controller = std::make_shared<CONTROLLER_T>(dynamics, cost, fb_controller, sampler, controller_params, stream);
    controller->setLogLevel(mppi::util::LOG_LEVEL::DEBUG);
    controller->chooseAppropriateKernel();

    plant = new PLANT_T(controller, 1.0f / settings.dt, settings.optimization_stride);
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

// using DIFFERENT_CONTROLLERS =
//     ::testing::Types<CONTROLLER_TEMPLATE<512>, CONTROLLER_TEMPLATE<2048>, CONTROLLER_TEMPLATE<8192>>;
// using AUTORALLY_CONTROLLERS =
//     ::testing::Types<AUTORALLY_MPPI_TEMPLATE<512>, AUTORALLY_MPPI_TEMPLATE<2048>, AUTORALLY_MPPI_TEMPLATE<8192>>;
using DIFFERENT_CONTROLLERS =
    ::testing::Types<CONTROLLER_TEMPLATE<128>, CONTROLLER_TEMPLATE<256>, CONTROLLER_TEMPLATE<512>,
                     CONTROLLER_TEMPLATE<1024>, CONTROLLER_TEMPLATE<2048>, CONTROLLER_TEMPLATE<4096>,
                     CONTROLLER_TEMPLATE<6144>, CONTROLLER_TEMPLATE<8192>, CONTROLLER_TEMPLATE<16384>>;
using AUTORALLY_CONTROLLERS =
    ::testing::Types<AUTORALLY_MPPI_TEMPLATE<128>, AUTORALLY_MPPI_TEMPLATE<256>, AUTORALLY_MPPI_TEMPLATE<512>,
                     AUTORALLY_MPPI_TEMPLATE<1024>, AUTORALLY_MPPI_TEMPLATE<2048>, AUTORALLY_MPPI_TEMPLATE<4096>,
                     AUTORALLY_MPPI_TEMPLATE<6144>, AUTORALLY_MPPI_TEMPLATE<8192>, AUTORALLY_MPPI_TEMPLATE<16384>>;

TYPED_TEST_SUITE(MPPIGenericAutorallyTest, DIFFERENT_CONTROLLERS);

TYPED_TEST(MPPIGenericAutorallyTest, AutorallyOnMPPIGeneric)
{
  RunningStats<double> times;
  std::atomic<bool> alive(true);
  for (int t = 0; t < this->settings.num_iterations; t++)
  {
    auto start = std::chrono::steady_clock::now();
    this->plant->updateState(this->plant->current_state_, t * this->settings.dt);
    this->plant->runControlIteration(&alive);
    auto end = std::chrono::steady_clock::now();
    double duration = (end - start).count() / 1e6;
    times.add(duration);
  }
  // Save to CSV File
  CSVWritingEnvironment::csv_file << CSVWritingEnvironment::getCPUName()
      << "," << CSVWritingEnvironment::getGPUName() << ",MPPI-Generic,"
      << this->controller->sampler_->getNumRollouts() << "," << times.mean()
      << "," << sqrt(times.variance()) << std::endl;
  printf("MPPI-Generic MPPI with %d rollouts optimization time: %f +- %f ms\n",
         this->controller->sampler_->getNumRollouts(), times.mean(), sqrt(times.variance()));
  printf("\tAverage Optimization Hz: %f Hz\n", 1000.0 / times.mean());
}


template <class CONTROLLER_T>
class AutorallyMPPITest : public ::testing::Test
{
public:

protected:
  AutorallySettings settings;

  std::shared_ptr<CONTROLLER_T> controller;
  DYN_T* dynamics = nullptr;
  COST_T* cost = nullptr;
  cudaStream_t stream;

  void SetUp() override
  {
    /**
     * Set up dynamics
     **/
    dynamics = new DYN_T();
    dynamics->loadParams(mppi_generic_testing::nn_file);
    std::array<float2, DYN_T::CONTROL_DIM> control_ranges;
    control_ranges[C_IND_CLASS(DYN_T::DYN_PARAMS_T, STEERING)] = settings.steering_range;
    control_ranges[C_IND_CLASS(DYN_T::DYN_PARAMS_T, THROTTLE)] = settings.throttle_range;
    dynamics->setControlRanges(control_ranges);

    /**
     * Set up Cost function
     **/
    cost = new COST_T();
    auto cost_params = cost->getParams();
    cost_params.speed_coeff = settings.speed_coeff;
    cost_params.track_coeff = settings.track_coeff;
    cost_params.slip_coeff = settings.slip_coeff;
    cost_params.crash_coeff = settings.crash_coeff;
    cost_params.boundary_threshold = settings.boundary_threshold;
    cost_params.desired_speed = settings.desired_speed;
    cost_params.max_slip_ang = settings.max_slip_angle;
    cost_params.track_slop = settings.track_slop;
    cost_params.discount = settings.discount;

    cost->setParams(cost_params);


    /**
     * Set up Controller
     **/

    HANDLE_ERROR(cudaStreamCreate(&stream));
    dynamics->bindToStream(stream);
    cost->bindToStream(stream);
    dynamics->GPUSetup();
    // Setup cost map
    cost->GPUSetup();
    cost->loadTrackData(mppi_generic_testing::track_file);
    dynamics->paramsToDevice();
    cost->paramsToDevice();
    cudaStreamSynchronize(stream);

    DYN_T::control_array init_control;
    init_control[C_IND_CLASS(DYN_T::DYN_PARAMS_T, STEERING)] = settings.init_steering;
    init_control[C_IND_CLASS(DYN_T::DYN_PARAMS_T, THROTTLE)] = settings.init_throttle;
    float variance[DYN_T::CONTROL_DIM];
    variance[C_IND_CLASS(DYN_T::DYN_PARAMS_T, STEERING)] = settings.std_steering;
    variance[C_IND_CLASS(DYN_T::DYN_PARAMS_T, THROTTLE)] = settings.std_throttle;
    controller = std::make_shared<CONTROLLER_T>(dynamics, cost, NUM_TIMESTEPS, 1 / settings.dt, 1 / settings.lambda,
                                                variance, init_control.data(), 1, 1, stream);
  }

  void TearDown() override
  {
    controller.reset();
    delete cost;
    delete dynamics;
  }
};


TYPED_TEST_SUITE(AutorallyMPPITest, AUTORALLY_CONTROLLERS);

TYPED_TEST(AutorallyMPPITest, AutorallyOnAutorally)
{
  RunningStats<double> times;
  std::atomic<bool> alive(true);
  DYN_T::state_array state = DYN_T::state_array::Zero();
  DYN_T::state_array prev_state, state_der;
  DYN_T::output_array output;
  DYN_T::control_array u;
  float dt = this->settings.dt;
  int query_point = 1;
  for (int t = 0; t < this->settings.num_iterations; t++)
  {
    auto start = std::chrono::steady_clock::now();
    this->controller->computeControl(state);
    auto u_traj = this->controller->getControlSeq();
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
      u[i] = u_traj[query_point * DYN_T::CONTROL_DIM + i];
    }
    prev_state = state;
    this->dynamics->step(prev_state, state, state_der, u, output, t * dt, dt);
    auto end = std::chrono::steady_clock::now();
    double duration = (end - start).count() / 1e6;
    times.add(duration);
  }
  // Save to CSV File
  CSVWritingEnvironment::csv_file << CSVWritingEnvironment::getCPUName()
      << "," << CSVWritingEnvironment::getGPUName() << ",autorally,"
      << this->controller->NUM_ROLLOUTS << "," << times.mean()
      << "," << sqrt(times.variance()) << std::endl;
  printf("Autorally MPPI with %d rollouts optimization time: %f +- %f ms\n",
         this->controller->NUM_ROLLOUTS, times.mean(), sqrt(times.variance()));
  printf("\tAverage Optimization Hz: %f Hz\n", 1000.0 / times.mean());
}
