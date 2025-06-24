/**
 * Created by Bogdan on 09/05/2023
 */
#pragma once
#include <mppi/cost_functions/cost.cuh>
#include <mppi/utils/texture_helpers/two_d_texture_helper.cuh>

struct GoalParams
{
  float2 pos;  // ([m], [m])
  float weight = 1.0f;
  float power = 1.0f;
};

struct GoalAngleParams
{
  float yaw;  // [rad]
  float weight = 1.0f;
  float power = 1.0f;
};

struct ObstacleParams
{
  float scale_factor = 1.0f;
  float min_radius = 0.1f;                 // [m]
  float inflation_radius = 0.1f;           // [m]
  float near_goal_distance = 0.5f;         // [m]
  float collision_margin_distance = 0.1f;  // [m]
  float collision_cost = 10000.0f;
  float traj_weight = 1.0f;
  float repulsion_weight = 0.0f;
  float power = 1.0f;
  bool use_footprint = false;
  float LETHAL_OBSTACLE = 1e10f;
};

template <class DYN_PARAMS_T>
struct ComparisonParams : public CostParams<C_IND_CLASS(DYN_PARAMS_T, NUM_CONTROLS)>
{
  float obstacle_cost = 1.0f;
  float goal_distance_threshold = 1000.0f;  // [m]
  int num_cosine_ops = 0;
  GoalParams goal;
  GoalAngleParams goal_angle;
  ObstacleParams obstacle;
};

template <class DYN_PARAMS_T>
class ComparisonCost : public Cost<ComparisonCost<DYN_PARAMS_T>, ComparisonParams<DYN_PARAMS_T>, DYN_PARAMS_T>
{
public:
  using PARENT_CLASS = Cost<ComparisonCost, ComparisonParams<DYN_PARAMS_T>, DYN_PARAMS_T>;
  using DYN_P = typename PARENT_CLASS::TEMPLATED_DYN_PARAMS;
  using output_array = typename PARENT_CLASS::output_array;
  static const int OBSTACLE_LAYER = 0;

  ComparisonCost(cudaStream_t stream = nullptr);

  void bindToStream(cudaStream_t stream)
  {
    if (tex_helper_)
    {
      tex_helper_->bindToStream(stream);
    }
    PARENT_CLASS::bindToStream(stream);
  }

  ~ComparisonCost();

  void GPUSetup();

  void paramsToDevice();

  void freeCudaMem();

  std::string getCostFunctionName() const override
  {
    return "ROS2 Comparison Cost";
  }

  float computeStateCost(const Eigen::Ref<const output_array> y, int timestep, int* crash_status)
  {
    return 0.0f;
  }

  float terminalCost(const Eigen::Ref<const output_array> y)
  {
    return 0.0f;
  }

  __host__ __device__ float computeStateCost(float* y, int timestep, float* theta_c, int* crash_status);

  __host__ __device__ float distanceToObstacle(float cost, ObstacleParams* p);

  __host__ __device__ float computeObstacleCost(const float* y, const int t, int* crash, float* theta_c);

  __host__ __device__ float computeGoalCost(const float* y, const int t, int* crash, float* theta_c);

  __host__ __device__ float computeGoalAngleCost(const float* y, const int t, int* crash, float* theta_c);

  __device__ float terminalCost(float* y, float* theta_c);

  TwoDTextureHelper<float>* tex_helper_ = nullptr;

protected:
};

#ifdef __CUDACC__
#include "comparison_cost.cu"
#endif
