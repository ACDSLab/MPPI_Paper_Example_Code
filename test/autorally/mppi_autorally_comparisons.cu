#include <gtest/gtest.h>
#include <mppi_paper_example/costs/ComparisonCost/comparison_cost.cuh>
#include <mppi_paper_example/dynamics/diff_drive/diff_drive.cuh>
#include <mppi_paper_example/plants/sim_plant/sim_plant.hpp>

#include "test/common.hpp"
#include "test/mppi_controller.cuh"

using DYN_T = DiffDrive;
using COST_T = ComparisonCost<DYN_T::DYN_PARAMS_T>;
const int NUM_TIMESTEPS = CommonSettings::num_timesteps;
const int BDIM_X = 64;
const int BDIM_Y = DYN_T::STATE_DIM;

template <int NUM_ROLLOUTS = 128>
using CONTROLLER_TEMPLATE = autorally_control::MPPIController<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y>;

template <class CONTROLLER_T>
class MPPIGenericMPPITest : public ::testing::Test
{
public:

protected:
  CommonSettings settings;

  std::shared_ptr<CONTROLLER_T> controller;
  DYN_T* dynamics = nullptr;
  COST_T* cost = nullptr;
  cudaStream_t stream;

  void SetUp() override
  {
    float variance[DYN_T::CONTROL_DIM];
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
      variance[i] = settings.std_dev_v * settings.std_dev_v;
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


    /**
     * Set up Controller
     **/
    HANDLE_ERROR(cudaStreamCreate(&stream));
    dynamics->bindToStream(stream);
    cost->bindToStream(stream);
    dynamics->GPUSetup();
    cost->GPUSetup();
    dynamics->paramsToDevice();
    cost->paramsToDevice();
    cudaStreamSynchronize(stream);
    DYN_T::control_array init_control = DYN_T::control_array::Zero();
    controller = std::make_shared<CONTROLLER_T>(dynamics, cost, NUM_TIMESTEPS, 1 / settings.dt, settings.lambda, variance,
                                                init_control.data(), 1, 1, stream);
  }

  void TearDown() override
  {
    controller.reset();
    delete cost;
    delete dynamics;
  }
};

using DIFFERENT_CONTROLLERS =
    ::testing::Types<CONTROLLER_TEMPLATE<128>, CONTROLLER_TEMPLATE<256>, CONTROLLER_TEMPLATE<512>,
                     CONTROLLER_TEMPLATE<1024>, CONTROLLER_TEMPLATE<2048>, CONTROLLER_TEMPLATE<4096>,
                     CONTROLLER_TEMPLATE<6144>, CONTROLLER_TEMPLATE<8192>, CONTROLLER_TEMPLATE<16384>>;

TYPED_TEST_SUITE(MPPIGenericMPPITest, DIFFERENT_CONTROLLERS);

TYPED_TEST(MPPIGenericMPPITest, DifferentNumSamples)
{
  RunningStats<double> times;
  std::atomic<bool> alive(true);
  DYN_T::state_array state = DYN_T::state_array::Zero();
  for (int t = 0; t < this->settings.num_iterations; t++)
  {
    auto start = std::chrono::steady_clock::now();
    this->controller->computeControl(state);
    auto end = std::chrono::steady_clock::now();
    double duration = (end - start).count() / 1e6;
    times.add(duration);
  }
  printf("MPPI-Generic MPPI with %d rollouts optimization time: %f +- %f ms\n",
         this->controller->NUM_ROLLOUTS, times.mean(), sqrt(times.variance()));
  printf("\tAverage Optimization Hz: %f Hz\n", 1000.0 / times.mean());
}
