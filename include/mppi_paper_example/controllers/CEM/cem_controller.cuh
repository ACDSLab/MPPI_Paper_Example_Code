/**
 * @file cem_controller.cuh
 * @brief Cross Entropy Controller Class
 * @author Bogdan Vlahov
 * @version 0.0.1
 * @date 2024-03-28
 */
#pragma once

#include <mppi/controllers/controller.cuh>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

template <int STATE_DIM, int CONTROL_DIM, int MAX_T>
struct CEMParams : public ControllerParams<STATE_DIM, CONTROL_DIM, MAX_T> {
  float top_k_percentage = 0.10f;
};

template <class DYN_T, class COST_T, class FB_T, int MAX_T, int NUM_SAMPLES,
          class SAMPLING_T =
              ::mppi::sampling_distributions::GaussianDistribution<typename DYN_T::DYN_PARAMS_T>,
          class PARAMS_T = CEMParams<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, MAX_T>>
class CEMController : public Controller<DYN_T, COST_T, FB_T, SAMPLING_T, MAX_T, NUM_SAMPLES, PARAMS_T> {
public:
  // Setup type aliases
  using PARENT_CLASS = Controller<DYN_T, COST_T, FB_T, SAMPLING_T, MAX_T, NUM_SAMPLES, PARAMS_T>;
  using control_array = typename PARENT_CLASS::control_array;
  using control_trajectory = typename PARENT_CLASS::control_trajectory;
  using state_trajectory = typename PARENT_CLASS::state_trajectory;
  using state_array = typename PARENT_CLASS::state_array;
  using sampled_cost_traj = typename PARENT_CLASS::sampled_cost_traj;
  using FEEDBACK_GPU = typename PARENT_CLASS::TEMPLATED_FEEDBACK_GPU;
  /**
   *
   * Public member functions
   */
  // Constructor
  CEMController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, PARAMS_T& params,
                cudaStream_t stream = 0);

  std::string getControllerName() {
    return "Cross Entropy";
  };

  /**
   * computes a new control sequence
   * @param state starting position
   */
  void computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride = 1) override;

  /**
   * Slide the control sequence back n steps
   */
  void slideControlSequence(int optimization_stride) override;

  /**
   * Call a kernel to evaluate the sampled state trajectories for visualization
   * and debugging.
   */
  void calculateSampledStateTrajectories() override {
  }

  void calculateEliteSet(const float* costs, const int num_costs, const int elite_set_size,
                         std::vector<int>& elite_indices);

protected:
  std::vector<int> top_k_indices_;
};

#ifdef __CUDACC__
#include "cem_controller.cu"
#endif
