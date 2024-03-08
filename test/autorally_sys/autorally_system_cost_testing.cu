/**
 * @file autorally_system_cost_testing.cu
 * @brief Comparing Autorally MPPI versus MPPI-Generic to see when they achieve
 * similar times given a computationally-heavy cost function
 * @author Bogdan Vlahov
 * @version 0.0.1
 * @date 2024-03-06
 */
#include <chrono>
#include <gtest/gtest.h>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/dynamics/autorally/ar_nn_model.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi_paper_example/costs/AutorallyModifiedCost/autorally_modified_cost.cuh>
#include <mppi_paper_example/plants/sim_plant/sim_plant.hpp>

#include "test/common.hpp"
#include "test/mppi_controller.cuh"

#include <autorally_system_testing.h>

#include <type_traits>
#include <utility>

const int NUM_TIMESTEPS = AutorallySettings::num_timesteps;
using DYN_T = NeuralNetModel<7, 2, 3, 6, 32, 32, 4>;
using COST_T = ARModifiedCost;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;

template <int NUM_ROLLOUTS = 128>
using CONTROLLER_TEMPLATE = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;

template <int NUM_ROLLOUTS = 128>
using AUTORALLY_MPPI_TEMPLATE =
    autorally_control::MPPIController<DYN_T, COST_T, NUM_ROLLOUTS, AutorallySettings::DYN_BLOCK_X,
                                      AutorallySettings::DYN_BLOCK_Y>;

template <int NUM_ROLLOUTS = 128>
using COMBINED_CONTROLLER_TEMPLATE =
    std::pair<std::shared_ptr<CONTROLLER_TEMPLATE<NUM_ROLLOUTS>>, AUTORALLY_MPPI_TEMPLATE<NUM_ROLLOUTS>*>;

class CSVWritingEnvironment : public ::testing::Environment
{
public:
  static std::ofstream csv_file;
  static std::string cpu_name;
  static std::string gpu_name;
  ~CSVWritingEnvironment() override
  {
  }
  void SetUp() override
  {
    std::string custom_csv_header =
        "Processor,GPU,Method,Num. Cosines,Num Rollouts,Mean Optimization Time "
        "(ms), Std. Dev. Time (ms)\n";
    createNewCSVFile("autorally_sys_compexity_results", csv_file, custom_csv_header);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    gpu_name = std::string(deviceProp.name);

    // get CPU name
    cpu_name = getCPUModelName();
  }

  void TearDown() override
  {
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

class MPPIGenericVsAutorally : public ::testing::Test
{
public:
  using pair_128 = COMBINED_CONTROLLER_TEMPLATE<128>;
  using pair_256 = COMBINED_CONTROLLER_TEMPLATE<256>;
  using pair_512 = COMBINED_CONTROLLER_TEMPLATE<512>;
  using pair_1024 = COMBINED_CONTROLLER_TEMPLATE<1024>;
  using pair_2048 = COMBINED_CONTROLLER_TEMPLATE<2048>;
  using pair_4096 = COMBINED_CONTROLLER_TEMPLATE<4096>;
  using pair_8192 = COMBINED_CONTROLLER_TEMPLATE<8192>;
  using pair_16384 = COMBINED_CONTROLLER_TEMPLATE<16384>;
  using pair_32768 = COMBINED_CONTROLLER_TEMPLATE<32768>;

  using controller_tuple =
      std::tuple<pair_128, pair_256, pair_512, pair_1024, pair_2048, pair_4096, pair_8192, pair_16384, pair_32768, >;

  struct Results
  {
    double mppi_generic_mean_ms = 0.0f;
    double mppi_generic_var_ms = 0.0f;
    double autorally_mean_ms = 0.0f;
    double autorally_var_ms = 0.0f;
    int num_rollouts = -1;
  };

protected:
  AutorallySettings settings;
  // const int DYN_BLOCK_X = 64;
  // const int DYN_BLOCK_Y = DYN_T::STATE_DIM;
  // const int COST_BLOCK_X = 64;

  DYN_T* dynamics = nullptr;
  COST_T* cost = nullptr;
  FB_T* fb_controller = nullptr;
  SAMPLING_T* sampler = nullptr;
  cudaStream_t stream;
  Eigen::Matrix<float, DYN_T::CONTROL_DIM, NUM_TIMESTEPS> init_control_traj;
  controller_tuple controllers;
  mppi::util::MPPILoggerPtr logger;

  void SetUp() override
  {
    logger = std::make_shared<mppi::util::MPPILogger>(mppi::util::LOG_LEVEL::INFO);
    SAMPLING_T::SAMPLING_PARAMS_T sampler_params;
    sampler_params.std_dev[C_IND_CLASS(DYN_T::DYN_PARAMS_T, STEERING)] = settings.std_steering;
    sampler_params.std_dev[C_IND_CLASS(DYN_T::DYN_PARAMS_T, THROTTLE)] = settings.std_throttle;
    // sampler_params.rewrite_controls_block_dim.y = 8;
    sampler = new SAMPLING_T(sampler_params);
    DYN_T::control_array u;
    u[C_IND_CLASS(DYN_T::DYN_PARAMS_T, STEERING)] = settings.init_steering;
    u[C_IND_CLASS(DYN_T::DYN_PARAMS_T, THROTTLE)] = settings.init_throttle;
    for (int i = 0; i < NUM_TIMESTEPS; i++)
    {
      init_control_traj.col(i) = u;
    }

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

    HANDLE_ERROR(cudaStreamCreate(&stream));
  }

  void TearDown() override
  {
    logger->debug("In the teardown\n");
    std::cout << std::endl;
    delete fb_controller;
    delete cost;
    delete dynamics;
    delete sampler;
  }

  // Create tuple method end
  template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), Results>::type testRollout(std::tuple<Tp...>& t)
  {
    Results empty_result;
    return empty_result;
  }

  // Create intermediate tuple method
  template <std::size_t I = 0, typename... Tp>
      inline typename std::enable_if < I<sizeof...(Tp), Results>::type testRollout(std::tuple<Tp...>& t)
  {
    // Get typing sorted out
    auto pair = std::get<I>(t);
    using MPPI_TYPE = typename decltype(pair)::first_type::element_type;
    using CONTROLLER_PARAMS_T = typename MPPI_TYPE::TEMPLATED_PARAMS;
    using PLANT_T = SimpleDynPlant<MPPI_TYPE>;
    using AUTORALLY_TYPE = typename std::remove_pointer<typename decltype(pair)::second_type>::type;

    /**
     * Set up Controller
     **/
    CONTROLLER_PARAMS_T controller_params;
    controller_params.dt_ = settings.dt;
    controller_params.lambda_ = settings.lambda;
    controller_params.dynamics_rollout_dim_ = dim3(settings.DYN_BLOCK_X, settings.DYN_BLOCK_Y, 1);
    controller_params.cost_rollout_dim_ = dim3(settings.COST_BLOCK_X, 1, 1);
    controller_params.num_iters_ = settings.num_iters;
    // controller_params.optimization_stride_ = settings.optimization_stride;

    /**
     * Create both controllers types
     */
    // First item in pair is MPPI-Generic and second is AutorallyMPPI
    logger->debug("Starting to make MPPI-Generic controller\n");
    pair.first = std::make_shared<MPPI_TYPE>(this->dynamics, this->cost, this->fb_controller, this->sampler,
                                             controller_params, this->stream);

    float std_dev[DYN_T::CONTROL_DIM] = { 0.0f };
    auto sampler_params = this->sampler->getParams();
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
      std_dev[i] = sampler_params.std_dev[i];
    }
    logger->debug("Starting to make Autorally controller\n");
    pair.second =
        new AUTORALLY_TYPE(this->dynamics, this->cost, NUM_TIMESTEPS, this->settings.dt, 1 / this->settings.lambda,
                           std_dev, this->init_control_traj.col(0).data(), 1, 1, this->stream);

    logger->debug("Making Plant\n");
    std::shared_ptr<PLANT_T> plant =
        std::make_shared<PLANT_T>(pair.first, 1.0f / this->settings.dt, this->settings.optimization_stride);
    plant->setLogger(logger);
    // Ensure sampler is reset
    this->sampler->copyImportanceSamplerToDevice(this->init_control_traj.data(), 0, true);

    RunningStats<double> times_generic, times_autorally;
    std::atomic<bool> alive(true);
    DYN_T::state_array initial_state = plant->current_state_;

    // Run MPPI-Generic
    logger->info("Starting %6d MPPI-Generic iterations\n", AUTORALLY_TYPE::NUM_ROLLOUTS);
    for (int t = 0; t < this->settings.num_iterations; t++)
    {
      auto start = std::chrono::steady_clock::now();
      plant->updateState(plant->current_state_, t * this->settings.dt);
      plant->runControlIteration(&alive);
      auto end = std::chrono::steady_clock::now();
      double duration_ms = (end - start).count() / 1e6;
      times_generic.add(duration_ms);
    }

    // Run Autorally Controller
    DYN_T::state_array state = initial_state;
    DYN_T::state_array prev_state, state_der;
    DYN_T::output_array output;
    DYN_T::control_array u;
    float dt = this->settings.dt;
    int query_point = 1;
    logger->info("Starting %6d Autorally iterations\n", AUTORALLY_TYPE::NUM_ROLLOUTS);
    for (int t = 0; t < this->settings.num_iterations; t++)
    {
      auto start = std::chrono::steady_clock::now();
      pair.second->computeControl(state);
      auto u_traj = pair.second->getControlSeq();
      for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
      {
        u[i] = u_traj[query_point * DYN_T::CONTROL_DIM + i];
      }
      prev_state = state;
      this->dynamics->step(prev_state, state, state_der, u, output, t * dt, dt);
      auto end = std::chrono::steady_clock::now();
      double durations_ms = (end - start).count() / 1e6;
      times_autorally.add(durations_ms);
    }

    // Calculate cross over point check
    double generic_average_time_ms = times_generic.mean();
    double autorally_average_time_ms = times_autorally.mean();
    double percent_diff = abs(generic_average_time_ms - autorally_average_time_ms) / generic_average_time_ms;
    logger->info(
        "%5d samples, MPPI-Generic times: %s%8.5f%s ms, Autorally "
        "times: %s%8.5f%s ms, relative percent: %f%%\n",
        AUTORALLY_TYPE::NUM_ROLLOUTS, mppi::util::YELLOW, generic_average_time_ms, mppi::util::CYAN, mppi::util::YELLOW,
        autorally_average_time_ms, mppi::util::CYAN, 100.0f * percent_diff);

    // Cleanup
    plant.reset();
    // pair.first.reset();
    // delete plant;
    // delete pair.first;
    delete pair.second;

    // If they are within 5% of each other, consider that the cross over point
    if (percent_diff < 0.05)
    {
      Results full_result;
      full_result.mppi_generic_mean_ms = generic_average_time_ms;
      full_result.mppi_generic_var_ms = times_generic.variance();
      full_result.autorally_mean_ms = autorally_average_time_ms;
      full_result.autorally_var_ms = times_autorally.variance();
      full_result.num_rollouts = AUTORALLY_TYPE::NUM_ROLLOUTS;
      return full_result;
    }
    else
    {
      return testRollout<I + 1, Tp...>(t);
    }
  }
};

TEST_F(MPPIGenericVsAutorally, EqualityPoint)
{
  std::vector<int> num_cosine_operations_vec = { 0, 5, 10, 15, 20, 25, 50 };
  this->logger->debug("Running GPU Setup\n");
  auto cost_params = this->cost->getParams();
  for (const auto& num_cos_op : num_cosine_operations_vec)
  {
    cost_params.num_cosine_ops = num_cos_op;
    this->cost->GPUSetup();
    this->cost->setParams(cost_params);
    this->cost->loadTrackData(mppi_generic_testing::track_file);
    auto result = this->testRollout(this->controllers);
    this->logger->warning("Cost with %3d cos() are nearly equivalent at %6d samples.\n", cost_params.num_cosine_ops,
                          result.num_rollouts);
  }
}
