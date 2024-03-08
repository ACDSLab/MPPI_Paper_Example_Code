inline __device__ float ARModifiedCost::computeStateCost(float* s, int timestep, float* theta_c, int* crash_status)
{
  float track_cost = this->getTrackCost(s, crash_status);
  float speed_cost = this->getSpeedCost(s, crash_status);
  // printf("speed %f\n", speed_cost);
  float stabilizing_cost = this->getStabilizingCost(s, crash_status);
  float crash_cost = powf(this->params_.discount, timestep) * this->getCrashCost(s, crash_status, timestep);
  float cost = 10;
  for (int i = 0; i < this->params_.num_cosine_ops; i++)
  {
    cost = cosf(cost);
  }
  cost *= 0.0f;
  cost = speed_cost + crash_cost + track_cost + stabilizing_cost;
  if (cost > MAX_COST_VALUE || isnan(cost))
  {  // TODO Handle max cost value in a generic way
    cost = MAX_COST_VALUE;
  }
  return cost;
}
