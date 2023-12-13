/**
 * Created By Bogdan on 12/12/2023
 */
#pragma once

#include <mppi/controllers/MPPI/mppi_controller.cuh>

template <int S_DIM, int C_DIM, int MAX_TIMESTEPS>
struct DMDMPPIParams : public ControllerParams<S_DIM, C_DIM, MAX_TIMESTEPS>
{
  float step_size = 0.8f;
  DMDMPPIParams() = default;
};

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
          class SAMPLING_T = ::mppi::sampling_distributions::GaussianDistribution<typename DYN_T::DYN_PARAMS_T>,
          class PARAMS_T = DMDMPPIParams<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, MAX_TIMESTEPS>>
class DMDMPPIController : public VanillaMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // nAeed control_array = ... so that we can initialize
  // Eigen::Matrix with Eigen::Matrix::Zero();
  typedef VanillaMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T> PARENT_CLASS;
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
  DMDMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, float dt, int max_iter,
                        float lambda, float alpha, int num_timesteps = MAX_TIMESTEPS,
                        const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
                        cudaStream_t stream = nullptr);
  DMDMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, PARAMS_T& params,
                        cudaStream_t stream = nullptr);

  // Destructor
  ~DMDMPPIController();

  std::string getControllerName()
  {
    return "Dynamic Mirror Descent MPPI";
  };

  void computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride = 1) override;
protected:
  control_trajectory previous_mean_ = control_trajectory::Zero();
};

#if __CUDACC__
#include "dmd_mppi_controller.cu"
#endif
