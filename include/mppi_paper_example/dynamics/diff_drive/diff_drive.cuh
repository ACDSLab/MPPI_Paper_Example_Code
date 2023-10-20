#pragma once
#include <mppi/dynamics/dynamics.cuh>

using namespace MPPI_internal;

#define FASTER_DYN_COMPUTATIONS

struct DiffDriveParams : public DynamicsParams
{
  enum class StateIndex : int
  {
    X = 0,
    Y,
    YAW,
    NUM_STATES
  };

  enum class ControlIndex : int
  {
    LEFT_ROT_SPD = 0,
    RIGHT_ROT_SPD,
    NUM_CONTROLS
  };

  enum class OutputIndex : int
  {
    X = 0,
    Y,
    YAW,
#ifdef FASTER_DYN_COMPUTATIONS
    FILLER,
#endif
    NUM_OUTPUTS
  };
  float r = 1.0f;
  float L = 1.0f;
};

class DiffDrive : public Dynamics<DiffDrive, DiffDriveParams>
{
public:
  using PARENT_CLASS = Dynamics<DiffDrive, DiffDriveParams>;

  std::string getDynamicsModelName() const override
  {
    return "DiffDrive";
  }

  DiffDrive(cudaStream_t stream = nullptr) : PARENT_CLASS(stream)
  {
  }

  void computeStateDeriv(const Eigen::Ref<const state_array>& x, const Eigen::Ref<const control_array>& u,
                         Eigen::Ref<state_array> x_dot);

  __device__ inline void computeStateDeriv(float* x, float* u, float* x_dot, float* theta_s);

#ifdef FASTER_DYN_COMPUTATIONS
  __device__  void step(float* x, float* x_next, float* xdot,
                        float* u, float* output, float* theta_s, const float t,
                        const float dt);

  using PARENT_CLASS::step;
#endif

  state_array stateFromMap(const std::map<std::string, float>& map) override;
};

#ifdef __CUDACC__
#include "diff_drive.cu"
#endif
