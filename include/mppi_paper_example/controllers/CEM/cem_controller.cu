#define CEM_TEMPLATE                                                                                         \
  template <class DYN_T, class COST_T, class FB_T, int MAX_T, int NUM_SAMPLES, class SAMPLING_T,             \
            class PARAMS_T>

#define CEM CEMController<DYN_T, COST_T, FB_T, MAX_T, NUM_SAMPLES, SAMPLING_T, PARAMS_T>

// clang-format off
CEM_TEMPLATE
CEM::CEMController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, PARAMS_T& params,
                   cudaStream_t stream) : PARENT_CLASS(model, cost, fb_controller, sampler, params, stream) {
  this->allocateCUDAMemoryHelper();
}
// clang-format on

CEM_TEMPLATE
void CEM::slideControlSequence(int optimization_stride) {
  this->saveControlHistoryHelper(optimization_stride, this->control_, this->control_history_);
  this->slideControlSequenceHelper(optimization_stride, this->control_);
}

CEM_TEMPLATE
void CEM::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride) {
  // Send the initial condition to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));
  for (int opt_iter = 0; opt_iter < this->getNumIters(); opt_iter++) {
    // Send the nominal control to the device
    this->copyNominalControlToDevice(false);

    // Generate noise data
    this->sampler_->generateSamples(optimization_stride, opt_iter, this->gen_, false);

    // Calculate state trajectories and costs from sampled control trajectories
    mppi::kernels::launchSplitRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_->model_d_, this->cost_->cost_d_, this->sampler_->sampling_d_, this->getDt(),
        this->getNumTimesteps(), NUM_SAMPLES, this->getLambda(), this->getAlpha(), this->initial_state_d_,
        this->output_d_, this->trajectory_costs_d_, this->params_.dynamics_rollout_dim_,
        this->params_.cost_rollout_dim_, this->stream_, false);

    // Copy the costs back to the host
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Setup vector to hold top k weight indices
    int top_k_to_keep = NUM_SAMPLES * this->params_.top_k_percentage;
    calculateEliteSet(this->trajectory_costs_, NUM_SAMPLES, top_k_to_keep, top_k_indices_);

    // keep weights of the elite set
    float min_elite_value = this->trajectory_costs_[top_k_indices_.back()];
    std::replace_if(
        this->trajectory_costs_.data(), this->trajectory_costs_.data() + NUM_SAMPLES,
        [min_elite_value](float cost) { return cost < min_elite_value; }, 0.0f);

    // Copy weights back to device
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_d_, this->trajectory_costs_.data(),
                                 NUM_SAMPLES * sizeof(float), cudaMemcpyHostToDevice, this->stream_));

    // Compute the normalizer
    this->setNormalizer(mppi_common::computeNormalizer(this->trajectory_costs_.data(), NUM_SAMPLES));

    // Calculate the new mean
    this->sampler_->updateDistributionParamsFromDevice(this->trajectory_costs_d_, this->getNormalizerCost(),
                                                       0, false);

    // Transfer the new control back to the host
    this->sampler_->setHostOptimalControlSequence(this->control_.data(), 0, true);
  }
  // Calculate optimal state and output trajectory from the current state and optimal control
  this->computeOutputTrajectoryHelper(this->output_, this->state_, state, this->control_);
}

CEM_TEMPLATE
void CEM::calculateEliteSet(const float* costs, const int num_costs, const int elite_set_size,
                            std::vector<int>& elite_indices) {
  elite_indices.resize(elite_set_size);
  std::fill(elite_indices.begin(), elite_indices.end(), -1);
  // Find top k sampled weights
  for (int i = 0; i < num_costs; i++) {
    for (int k = 0; k < elite_set_size; k++) {
      if (elite_indices[k] == -1) {  // Empty spot, fill in with i
        elite_indices[k] = i;
        break;
      } else if (costs[i] > costs[elite_indices[k]]) {
        // ith value is larger than the kth largest value, shift smaller top k down & store i
        for (int j = elite_set_size - 1; j > k; j--) {
          elite_indices[j] = elite_indices[j - 1];
        }
        elite_indices[k] = i;
        break;
      }
    }
  }
}

#undef CEM_TEMPLATE
#undef CEM
