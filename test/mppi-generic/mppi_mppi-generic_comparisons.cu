#include <gtest/gtest.h>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi_paper_example/costs/ComparisonCost/comparison_cost.cuh>
#include <mppi_paper_example/dynamics/diff_drive/diff_drive.cuh>
#include <mppi_paper_example/plants/sim_plant/sim_plant.hpp>

#include "test/common.hpp"

#define USE_NEW_API

const int NUM_TIMESTEPS = CommonSettings::num_timesteps;
using DYN_T = DiffDrive;
using COST_T = ComparisonCost<DYN_T::DYN_PARAMS_T>;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;

template <int NUM_ROLLOUTS = 128>
using CONTROLLER_TEMPLATE = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;

template <class CONTROLLER_T>
class MPPIGenericMPPITest : public ::testing::Test
{
public:
  using CONTROLLER_PARAMS_T = typename CONTROLLER_T::TEMPLATED_PARAMS;
  using PLANT_T = SimpleDynPlant<CONTROLLER_T>;
protected:
  CommonSettings settings;
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
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
      sampler_params.std_dev[i] = 1.0f;
    }
    sampler = new SAMPLING_T(sampler_params);

    /**
     * Set up dynamics
     **/
    dynamics = new DYN_T();

    /**
     * Set up Cost function
     **/
    cost = new COST_T();
    auto cost_params = cost->getParams();
    cost_params.goal.pos = make_float2(settings.goal_x, settings.goal_y);
    cost_params.goal_angle.yaw = settings.goal_yaw;
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

    /**
     * Set up Feedback Controller
     **/
    fb_controller = new FB_T(dynamics, settings.dt);

    /**
     * Set up Controller
     **/
    controller_params.dt_ = settings.dt;
    controller_params.lambda_ = settings.lambda;
    controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1);
    controller_params.cost_rollout_dim_ = dim3(COST_BLOCK_X, 1, 1);

    HANDLE_ERROR(cudaStreamCreate(&stream));

    controller = std::make_shared<CONTROLLER_T>(dynamics, cost, fb_controller, sampler, controller_params, stream);

    plant = new PLANT_T(controller, 1.0f / settings.dt, 1);
  }

  void TearDown() override
  {
    delete plant;
    controller.reset();

  }
};

using DIFFERENT_CONTROLLERS = ::testing::Types<CONTROLLER_TEMPLATE<128>,
      CONTROLLER_TEMPLATE<256>, CONTROLLER_TEMPLATE<512>, CONTROLLER_TEMPLATE<1024>,
      CONTROLLER_TEMPLATE<2048>,
      CONTROLLER_TEMPLATE<4096>, CONTROLLER_TEMPLATE<6144>, CONTROLLER_TEMPLATE<8192>,
      CONTROLLER_TEMPLATE<4096*4>>;

TYPED_TEST_SUITE(MPPIGenericMPPITest, DIFFERENT_CONTROLLERS);

TYPED_TEST(MPPIGenericMPPITest, DifferentNumSamples)
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
  printf("MPPI-Generic MPPI with %d rollouts optimization time: %f +- %f ms\n",
         this->controller->sampler_->getNumRollouts(),
         times.mean(), sqrt(times.variance()));
  auto loop_time_ms = this->plant->getAvgLoopTime();
  std::cout << "Avg Loop time: " << loop_time_ms << " ms" << std::endl;
  std::cout << "Average Optimization Hz: " << 1.0f / (loop_time_ms * 1e-3f) << " Hz" << std::endl;
}
