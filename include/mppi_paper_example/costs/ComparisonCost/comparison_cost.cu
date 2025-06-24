/**
 * Created by Bogdan on 09/05/2023
 **/
#include "comparison_cost.cuh"
#include <mppi/utils/angle_utils.cuh>

#define COMPARISON_COST_TEMPLATE template <class DYN_PARAMS_T>
#define COMPARISON_COST ComparisonCost<DYN_PARAMS_T>

COMPARISON_COST_TEMPLATE
COMPARISON_COST::ComparisonCost(cudaStream_t stream)
{
  tex_helper_ = new TwoDTextureHelper<float>(1, stream);
  this->bindToStream(stream);
}

COMPARISON_COST_TEMPLATE
COMPARISON_COST::~ComparisonCost()
{
  delete tex_helper_;
}

COMPARISON_COST_TEMPLATE
void COMPARISON_COST::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    tex_helper_->copyToDevice();
  }
  PARENT_CLASS::paramsToDevice();
}

COMPARISON_COST_TEMPLATE
void COMPARISON_COST::GPUSetup()
{
  tex_helper_->GPUSetup();
  PARENT_CLASS* derived = static_cast<PARENT_CLASS*>(this);
  derived->GPUSetup();

  HANDLE_ERROR(cudaMemcpyAsync(&(this->cost_d_->tex_helper_), &(tex_helper_->ptr_d_), sizeof(TwoDTextureHelper<float>*),
                               cudaMemcpyHostToDevice, this->stream_));
}

COMPARISON_COST_TEMPLATE
void COMPARISON_COST::freeCudaMem()
{
  if (this->GPUMemStatus_)
  {
    tex_helper_->freeCudaMem();
  }
  PARENT_CLASS::freeCudaMem();
}

COMPARISON_COST_TEMPLATE
__host__ __device__ float COMPARISON_COST::computeStateCost(float* y, int t, float* theta_c, int* crash_status)
{
  float cost = 1.0f;
  for (int i = 0; i < this->params_.num_cosine_ops; i++)
  {
    cost = cosf(cost);
  }
  cost *= 0.0f;
  float obstacle_cost = computeObstacleCost(y, t, crash_status, theta_c);
  float goal_cost = computeGoalCost(y, t, crash_status, theta_c);
  float goal_angle_cost = computeGoalAngleCost(y, t, crash_status, theta_c);
  cost = obstacle_cost + goal_cost + goal_angle_cost;
  return cost;
}

COMPARISON_COST_TEMPLATE
__host__ __device__ float COMPARISON_COST::distanceToObstacle(float cost, ObstacleParams* p)
{
  // this expects a kernel based distance calculation stored in a char where 253
  // -> 1 -> on an object and a char value of 1 -> 1/253 -> distant from object
  float dist_to_obj = (p->scale_factor * p->min_radius - logf(cost) + logf(253.0f)) / p->scale_factor;

  if (!p->use_footprint)
  {
    dist_to_obj -= p->min_radius;
  }
  return dist_to_obj;
}

COMPARISON_COST_TEMPLATE
__host__ __device__ float COMPARISON_COST::computeObstacleCost(const float* y, const int t, int* crash, float* theta_c)
{
  if (!this->tex_helper_->checkTextureUse(OBSTACLE_LAYER))
  {
    return 0.0f;
  }
  ComparisonParams<DYN_PARAMS_T>* params_p = &this->params_;
  float cost = 0.0f;
  float3 query = make_float3(y[O_IND_CLASS(DYN_P, X)], y[O_IND_CLASS(DYN_P, Y)], 0.0f);
  const float dist_m = hypotf(query.x - params_p->goal.pos.x, query.y - params_p->goal.pos.y);
  bool near_goal = dist_m < params_p->obstacle.near_goal_distance;
  float tex_query = this->tex_helper_->queryTextureAtWorldPose(OBSTACLE_LAYER, query);

  if (tex_query < 1.0f)
  {  // in free space?
    return cost;
  }

  if (tex_query > params_p->obstacle.LETHAL_OBSTACLE || *crash)
  {  // lethal crash
    *crash = 1;
    return params_p->obstacle.collision_cost;
  }
  if (params_p->obstacle.scale_factor == 0.0f || params_p->obstacle.inflation_radius == 0.0f)
  {
    return cost;
  }
  const float dist_to_obj = distanceToObstacle(tex_query, &params_p->obstacle);
  if (dist_to_obj < params_p->obstacle.collision_margin_distance)
  {
    cost = params_p->obstacle.traj_weight * (params_p->obstacle.collision_margin_distance - dist_to_obj);
  }
  else if (!near_goal)
  {
    cost = params_p->obstacle.repulsion_weight * (params_p->obstacle.inflation_radius - dist_to_obj);
  }
  cost = powf(cost, params_p->obstacle.power);
  return cost;
}

COMPARISON_COST_TEMPLATE
__host__ __device__ float COMPARISON_COST::computeGoalCost(const float* y, const int t, int* crash, float* theta_c)
{
  ComparisonParams<DYN_PARAMS_T>* params_p = &this->params_;

  float cost = 0.0f;
  const float2 p = make_float2(y[O_IND_CLASS(DYN_P, X)], y[O_IND_CLASS(DYN_P, Y)]);

  const float dist_m = hypotf(p.x - params_p->goal.pos.x, p.y - params_p->goal.pos.y);
  if (dist_m < params_p->goal_distance_threshold)
  {
    cost = powf(dist_m * params_p->goal.weight, params_p->goal.power);
  }
  return cost;
}

COMPARISON_COST_TEMPLATE
__host__ __device__ float COMPARISON_COST::computeGoalAngleCost(const float* y, const int t, int* crash, float* theta_c)
{
  ComparisonParams<DYN_PARAMS_T>* params_p = &this->params_;

  float cost = 0.0f;
  const float2 p = make_float2(y[O_IND_CLASS(DYN_P, X)], y[O_IND_CLASS(DYN_P, Y)]);
  const float yaw = y[O_IND_CLASS(DYN_P, YAW)];

  const float dist_m = hypotf(p.x - params_p->goal.pos.x, p.y - params_p->goal.pos.y);
  const float dist_rad = angle_utils::shortestAngularDistance(yaw, params_p->goal_angle.yaw);
  if (dist_m < params_p->goal_distance_threshold)
  {
    cost = powf(fabsf(dist_rad) * params_p->goal_angle.weight, params_p->goal_angle.power);
  }
  return cost;
}

COMPARISON_COST_TEMPLATE
__device__ float COMPARISON_COST::terminalCost(float* y, float* theta_c)
{
  return 0.0f;
}
#undef COMPARISON_COST
#undef COMPARISON_COST_TEMPLATE
