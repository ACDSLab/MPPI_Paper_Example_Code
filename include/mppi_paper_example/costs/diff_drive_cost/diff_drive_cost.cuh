#pragma once
#include <mppi/cost_functions/cost.cuh>

#include <mppi_paper_example/dynamics/diff_drive/diff_drive.cuh>

struct DiffDriveCostParams : public CostParams<DiffDrive::CONTROL_DIM>
{
  float width = 1.0;
  float coeff = 10.0;
};

class DiffDriveCost : public Cost<DiffDriveCost, DiffDriveCostParams, DiffDrive::DYN_PARAMS_T>
{
public:
  using PARENT_CLASS = Cost<DiffDriveCost, DiffDriveCostParams, DiffDrive::DYN_PARAMS_T>;
  using DYN_P = PARENT_CLASS::TEMPLATED_DYN_PARAMS;

  DiffDriveCost(cudaStream_t stream = nullptr);

  std::string getCostFunctionName() const override
  {
    return "DiffDrive Cost";
  }

  float computeStateCost(const Eigen::Ref<const output_array> y, int timestep, int* crash_status);

  float terminalCost(const Eigen::Ref<const output_array> y);

  __device__ float computeStateCost(float* y, int timestep, float* theta_c, int* crash_status);

  __device__ float terminalCost(float* y, float* theta_c);
};

#ifdef __CUDACC__
#include "diff_drive_cost.cu"
#endif
