#pragma once
#include "diff_drive_cost.cuh"

DiffDriveCost::DiffDriveCost(cudaStream_t stream)
{
  this->bindToStream(stream);
}

float DiffDriveCost::computeStateCost(const Eigen::Ref<const output_array> y, int timestep, int* crash_status) {
  float cost = 0;
  float y_abs = abs(y[O_IND_CLASS(DYN_P, Y)]);
  if (y_abs < this->params_.width) { // Linear cost within the road width
    cost = y_abs;
  } else { // Quadratic cost outside the road width
    cost = y_abs * y_abs;
  }
  return this->params_.coeff * cost;
}

float DiffDriveCost::terminalCost(const Eigen::Ref<const output_array> y) {
  return 0;
}

__device__ float DiffDriveCost::computeStateCost(float* y, int timestep, float* theta_c, int* crash_status) {
  float cost = 0;
  float y_abs = abs(y[O_IND_CLASS(DYN_P, Y)]);
  if (y_abs < this->params_.width) { // Linear cost within the road width
    cost = y_abs;
  } else { // Quadratic cost outside the road width
    cost = y_abs * y_abs;
  }
  return this->params_.coeff * cost;
}

__device__ float DiffDriveCost::terminalCost(float* y, float* theta_c) {
  return 0;
}
