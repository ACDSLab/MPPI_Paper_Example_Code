#pragma once
#include <mppi/core/base_plant.hpp>
#include <mppi/dynamics/cartpole/cartpole_dynamics.cuh>

template <class CONTROLLER_T>
class SimpleCartpolePlant : public BasePlant<CONTROLLER_T>
{
public:
  using control_array = typename CartpoleDynamics::control_array;
  using state_array = typename CartpoleDynamics::state_array;
  using output_array = typename CartpoleDynamics::output_array;

  SimpleCartpolePlant(std::shared_ptr<CONTROLLER_T> controller, int hz, int optimization_stride)
    : BasePlant<CONTROLLER_T>(controller, hz, optimization_stride)
  {
    system_dynamics_ = std::make_shared<CartpoleDynamics>(1.0, 1.0, 1.0);
  }

  void pubControl(const control_array& u)
  {
    state_array state_derivative;
    output_array dynamics_output;
    state_array prev_state = current_state_;
    float t = this->state_time_;
    float dt = this->controller_->getDt();
    system_dynamics_->step(prev_state, current_state_, state_derivative, u, dynamics_output, t, dt);
    current_time_ += dt;
  }

  void pubNominalState(const state_array& s)
  {
  }

  void pubFreeEnergyStatistics(MPPIFreeEnergyStatistics& fe_stats)
  {
  }

  int checkStatus()
  {
    return 0;
  }

  double getCurrentTime()
  {
    return current_time_;
  }

  double getPoseTime()
  {
    return this->state_time_;
  }

  double getAvgLoopTime() const
  {
    return this->avg_loop_time_ms_;
  }

  double getLastOptimizationTime() const
  {
    return this->optimization_duration_;
  }

  state_array current_state_ = state_array::Zero();

protected:
  std::shared_ptr<CartpoleDynamics> system_dynamics_;
  double current_time_ = 0.0;
};
