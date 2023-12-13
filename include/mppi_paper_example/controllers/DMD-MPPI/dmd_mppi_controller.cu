#include <mppi_paper_example/controllers/DMD-MPPI/dmd_mppi_controller.cuh>

#define DMD_MPPI_TEMPLATE                                                                                          \
  template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, class SAMPLING_T,              \
            class PARAMS_T>

#define DMDMPPI DMDMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>

DMD_MPPI_TEMPLATE
DMDMPPI::DMDMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, float dt,
                           int max_iter, float lambda, float alpha, int num_timesteps,
                           const Eigen::Ref<const control_trajectory>& init_control_traj, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, dt, max_iter, lambda, alpha, num_timesteps, init_control_traj,
                 stream)
{
  previous_mean_ = init_control_traj;
}

DMD_MPPI_TEMPLATE
DMDMPPI::DMDMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler,
                                   PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, params, stream)
{
  previous_mean_ = params.init_control_traj_;
}

DMD_MPPI_TEMPLATE
DMDMPPI::~DMDMPPIController()
{

}

DMD_MPPI_TEMPLATE
void DMDMPPI::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride)
{
  // save previous mean
  previous_mean_ = this->control_;
  // calculate new control
  PARENT_CLASS::computeControl(state, optimization_stride);
  // Apply step size update
  this->control_ = (1.0f - this->params_.step_size) * previous_mean_ + this->params_.step_size * this->control_;
}
