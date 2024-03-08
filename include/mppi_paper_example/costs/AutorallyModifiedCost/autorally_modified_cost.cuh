#pragma once
/**
 * @file autorally_modified_cost.cuh
 * @brief Adds a variable number of cosf() calls to the autorally cost function
 * for testing purposes.
 * @author Bogdan Vlahov
 * @version 0.0.1
 * @date 2024-03-06
 */

#include <mppi/cost_functions/autorally/ar_standard_cost.cuh>

struct TestARStandardCostParams : public ARStandardCostParams
{
  int num_cosine_ops = 0;
};

class ARModifiedCost : public ARStandardCostImpl<ARModifiedCost, TestARStandardCostParams>
{
public:
  using PARENT_CLASS = ARStandardCostImpl<ARModifiedCost, TestARStandardCostParams>;

  ARModifiedCost(cudaStream_t stream = 0) : PARENT_CLASS(stream){};

  std::string getCostFunctionName() const override
  {
    return "AutoRally cost with " + std::to_string(this->params_.num_cosine_ops) + " cosines";
  }

  __device__ float computeStateCost(float* s, int timestep, float* theta_c, int* crash_status);
};

#ifdef __CUDACC__
#include "autorally_modified_cost.cu"
#endif
