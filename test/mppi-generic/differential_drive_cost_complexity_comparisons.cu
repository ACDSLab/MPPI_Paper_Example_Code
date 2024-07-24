/**
 * @file autorally_system_cost_testing.cu
 * @brief Comparing Autorally MPPI versus MPPI-Generic to see when they achieve
 * similar times given a computationally-heavy cost function
 * @author Bogdan Vlahov
 * @version 0.0.1
 * @date 2024-03-06
 */
#include <gtest/gtest.h>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi_paper_example/costs/ComparisonCost/comparison_cost.cuh>
#include <mppi_paper_example/dynamics/diff_drive/diff_drive.cuh>
#include <mppi_paper_example/plants/sim_plant/sim_plant.hpp>

#include "test/common.hpp"
#include "test/mppi_controller.cuh"

#include <autorally_system_testing.h>

#include <type_traits>
#include <utility>

const int NUM_TIMESTEPS = CommonSettings::num_timesteps;
using DYN_T = DiffDrive;
using COST_T = ComparisonCost<DYN_T::DYN_PARAMS_T>;
using FB_STATIC_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;

template <int NUM_ROLLOUTS = 128, int NUM_TIMESTEPS_T = CommonSettings::num_timesteps>
using CONTROLLER_TEMPLATE = VanillaMPPIController<DYN_T, COST_T, DDPFeedback<DYN_T, NUM_TIMESTEPS_T>, NUM_TIMESTEPS_T,
                                                  NUM_ROLLOUTS, SAMPLING_T>;

template <int NUM_ROLLOUTS = 128>
using AUTORALLY_MPPI_TEMPLATE =
    autorally_control::MPPIController<DYN_T, COST_T, NUM_ROLLOUTS, CommonSettings::DYN_BLOCK_X,
                                      CommonSettings::DYN_BLOCK_Y>;

template <int NUM_ROLLOUTS = 128, int NUM_TIMESTEPS_T = CommonSettings::num_timesteps>
using COMBINED_CONTROLLER_TEMPLATE = std::pair<std::shared_ptr<CONTROLLER_TEMPLATE<NUM_ROLLOUTS, NUM_TIMESTEPS_T>>,
                                               AUTORALLY_MPPI_TEMPLATE<NUM_ROLLOUTS>*>;

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
        "Processor,GPU,Method,Num. Cosines,Num Timesteps,Num Rollouts,Mean Optimization Time (ms), Std. Dev. Time "
        "(ms)\n";
    createNewCSVFile("diff_drive_compexity_results", csv_file, custom_csv_header);
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
  // clang-format off
  using controller_tuple = std::tuple<
      COMBINED_CONTROLLER_TEMPLATE<8192, CommonSettings::num_timesteps * 1>,
      COMBINED_CONTROLLER_TEMPLATE<8192, CommonSettings::num_timesteps * 2>,
      COMBINED_CONTROLLER_TEMPLATE<8192, CommonSettings::num_timesteps * 3>,
      COMBINED_CONTROLLER_TEMPLATE<8192, CommonSettings::num_timesteps * 4>,
      >;
  // clang-format on

  struct Results
  {
    double mppi_generic_mean_ms = 0.0f;
    double mppi_generic_var_ms = 0.0f;
    double autorally_mean_ms = 0.0f;
    double autorally_var_ms = 0.0f;
    int num_rollouts = -1;
  };

protected:
  CommonSettings settings;

  DYN_T* dynamics = nullptr;
  COST_T* cost = nullptr;
  SAMPLING_T* sampler = nullptr;
  cudaStream_t stream;
  Eigen::Matrix<float, DYN_T::CONTROL_DIM, NUM_TIMESTEPS> init_control_traj;
  controller_tuple controllers;
  mppi::util::MPPILoggerPtr logger;

  void SetUp() override
  {
    logger = std::make_shared<mppi::util::MPPILogger>(mppi::util::LOG_LEVEL::INFO);
    SAMPLING_T::SAMPLING_PARAMS_T sampler_params;
    sampler_params.std_dev[C_IND_CLASS(DYN_T::DYN_PARAMS_T, LEFT_ROT_SPD)] = settings.std_dev_v;
    sampler_params.std_dev[C_IND_CLASS(DYN_T::DYN_PARAMS_T, RIGHT_ROT_SPD)] = settings.std_dev_v;
    // sampler_params.rewrite_controls_block_dim.y = 8;
    sampler = new SAMPLING_T(sampler_params);
    DYN_T::control_array u;
    u[C_IND_CLASS(DYN_T::DYN_PARAMS_T, LEFT_ROT_SPD)] = 0.0f;
    u[C_IND_CLASS(DYN_T::DYN_PARAMS_T, RIGHT_ROT_SPD)] = 0.0f;
    for (int i = 0; i < NUM_TIMESTEPS; i++)
    {
      init_control_traj.col(i) = u;
    }

    /**
     * Set up dynamics
     **/
    dynamics = new DYN_T();
    auto dynamics_params = dynamics->getParams();
    dynamics_params.r = settings.robot_radius;
    dynamics_params.L = settings.robot_length;
    dynamics->setParams(dynamics_params);
    float2 control_range = make_float2(settings.v_min, settings.v_max);
    std::array<float2, DYN_T::CONTROL_DIM> control_ranges;
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
      control_ranges[i] = control_range;
    }
    dynamics->setControlRanges(control_ranges);

    /**
     * Set up Cost function
     **/
    cost = new COST_T();
    auto cost_params = cost->getParams();
    cost_params.goal.pos = make_float2(settings.goal_x, settings.goal_y);
    cost_params.goal_angle.yaw = settings.goal_yaw;
    cost_params.goal.power = settings.goal_power;
    cost_params.goal.weight = settings.goal_weight;
    cost_params.goal_angle.power = settings.goal_angle_power;
    cost_params.goal_angle.weight = settings.goal_angle_weight;
    cost_params.goal_distance_threshold = settings.goal_dist_threshold;
    cost_params.obstacle.use_footprint = settings.consider_footprint;
    cost_params.obstacle.near_goal_distance = settings.near_goal_distance;
    cost_params.obstacle.inflation_radius = settings.obs_inflation_radius;
    cost_params.obstacle.repulsion_weight = settings.obs_repulsion_weight;
    cost_params.obstacle.scale_factor = settings.obs_scaling_factor;
    cost_params.obstacle.traj_weight = settings.obs_traj_weight;
    cost_params.obstacle.power = settings.obs_power;
    cost_params.obstacle.min_radius = 0.0f;
    cost->setParams(cost_params);

    // Setup cost map
    int rows = settings.length_x / settings.resolution;
    int cols = settings.length_y / settings.resolution;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXfROW;
    MatrixXfROW map = MatrixXfROW::Constant(rows, cols, settings.costmap_default_val * 1.0f);

    // Add Obstacles
    for (int i = 0; i < settings.NUM_OBSTACLES; i++)
    {
      int obs_x = settings.obstacle_pos_x[i] / settings.resolution;
      int obs_y = settings.obstacle_pos_y[i] / settings.resolution;
      int obs_size = settings.obstacle_sizes[i] / settings.resolution;
      for (int row = obs_x; row < obs_x + obs_size; row++)
      {
        for (int col = obs_y; col < obs_y + obs_size; col++)
        {
          map(row, col) = settings.obstacle_cost * 1.0f;
        }
      }
    }

    cudaExtent extent = make_cudaExtent(cols, rows, 0);
    float3 origin = make_float3(settings.origin_x, settings.origin_y, 0.0f);
    cost->tex_helper_->updateOrigin(0, origin);
    cost->tex_helper_->updateResolution(0, settings.resolution);
    cost->tex_helper_->setExtent(0, extent);
    cost->tex_helper_->updateTexture(0, map, true);
    cost->tex_helper_->enableTexture(0);

    HANDLE_ERROR(cudaStreamCreate(&stream));
  }

  void TearDown() override
  {
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
    using FB_T = typename MPPI_TYPE::TEMPLATED_FEEDBACK;

    /**
     * Set up Controller
     **/
    std::shared_ptr<FB_T> fb_controller;
    fb_controller = std::make_shared<FB_T>(dynamics, settings.dt);

    CONTROLLER_PARAMS_T controller_params;
    controller_params.dt_ = settings.dt;
    controller_params.lambda_ = settings.lambda;
    controller_params.dynamics_rollout_dim_ = dim3(settings.DYN_BLOCK_X, settings.DYN_BLOCK_Y, 1);
    controller_params.cost_rollout_dim_ = dim3(settings.COST_BLOCK_X, 1, 1);
    controller_params.num_iters_ = settings.iteration_count;
    // controller_params.optimization_stride_ = settings.optimization_stride;

    /**
     * Create both controllers types
     */
    // First item in pair is MPPI-Generic and second is AutorallyMPPI
    logger->debug("Starting to make MPPI-Generic controller\n");
    pair.first = std::make_shared<MPPI_TYPE>(this->dynamics, this->cost, fb_controller.get(), this->sampler,
                                             controller_params, this->stream);

    // Get num_timesteps for autorally controller
    const int num_timesteps_t = pair.first->getNumTimesteps();

    float std_dev[DYN_T::CONTROL_DIM] = { 0.0f };
    auto sampler_params = this->sampler->getParams();
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
      std_dev[i] = sampler_params.std_dev[i];
    }
    logger->debug("Starting to make Autorally controller\n");
    pair.second =
        new AUTORALLY_TYPE(this->dynamics, this->cost, num_timesteps_t, this->settings.dt, 1 / this->settings.lambda,
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
      plant->updateState(plant->current_state_, (t + 1) * this->settings.dt);
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

    const int num_cos_op = cost->getParams().num_cosine_ops;
    logger->warning(
        "%5d samples, %4d timesteps, %3d cosines, MPPI-Generic times: %s%8.5f%s ms, Autorally times: %s%8.5f%s ms, "
        "relative percent: %f%%\n",
        AUTORALLY_TYPE::NUM_ROLLOUTS, num_timesteps_t, num_cos_op, mppi::util::GREEN, generic_average_time_ms,
        mppi::util::YELLOW, mppi::util::GREEN, autorally_average_time_ms, mppi::util::YELLOW, 100.0f * percent_diff);

    // Cleanup
    plant.reset();
    // pair.first.reset();
    delete pair.second;

    Results result;
    result.mppi_generic_mean_ms = generic_average_time_ms;
    result.mppi_generic_var_ms = times_generic.variance();
    result.autorally_mean_ms = autorally_average_time_ms;
    result.autorally_var_ms = times_autorally.variance();
    result.num_rollouts = AUTORALLY_TYPE::NUM_ROLLOUTS;

    // Fill in CSV file here
    CSVWritingEnvironment::csv_file << CSVWritingEnvironment::getCPUName() << "," << CSVWritingEnvironment::getGPUName()
                                    << ",MPPI-Generic," << num_cos_op << "," << num_timesteps_t << ","
                                    << result.num_rollouts << "," << result.mppi_generic_mean_ms << ","
                                    << sqrtf(result.mppi_generic_var_ms) << std::endl;
    CSVWritingEnvironment::csv_file << CSVWritingEnvironment::getCPUName() << "," << CSVWritingEnvironment::getGPUName()
                                    << ",autorally," << num_cos_op << "," << num_timesteps_t << ","
                                    << result.num_rollouts << "," << result.autorally_mean_ms << ","
                                    << sqrtf(result.autorally_var_ms) << std::endl;

    return testRollout<I + 1, Tp...>(t);
  }
};

TEST_F(MPPIGenericVsAutorally, EqualityPoint)
{
  std::vector<int> num_cosine_operations_vec;
  for (int i = 0; i < 16; i++)
  {
    num_cosine_operations_vec.push_back(i * 10);
  }
  this->logger->debug("Running GPU Setup\n");
  auto cost_params = this->cost->getParams();
  for (const auto& num_cos_op : num_cosine_operations_vec)
  {
    cost_params.num_cosine_ops = num_cos_op;
    // Need to recreate the track map every call
    this->cost->GPUSetup();
    this->cost->setParams(cost_params);
    auto result = this->testRollout(this->controllers);
  }
}
