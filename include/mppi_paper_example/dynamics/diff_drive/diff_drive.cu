#pragma once
#include "diff_drive.cuh"
#include "mppi/utils/angle_utils.cuh"

void DiffDrive::computeStateDeriv(
    const Eigen::Ref<const state_array>& x,
    const Eigen::Ref<const control_array>& u,
    Eigen::Ref<state_array> xdot) {
  // xdot[S_INDEX(X)] = u[C_INDEX(VEL)] * cosf(x[S_INDEX(YAW)]);
  // xdot[S_INDEX(Y)] = u[C_INDEX(VEL)] * sinf(x[S_INDEX(YAW)]);
  // xdot[S_INDEX(YAW)]   = u[C_INDEX(YAW_DOT)];
  float l_spd = u[C_INDEX(LEFT_ROT_SPD)];
  float r_spd = u[C_INDEX(RIGHT_ROT_SPD)];
  float yaw = x[S_INDEX(YAW)];
  xdot[S_INDEX(X)] = this->params_.r / 2.0f * (l_spd + r_spd) * cosf(yaw);
  xdot[S_INDEX(Y)] = this->params_.r / 2.0f * (l_spd + r_spd) * sinf(yaw);
  xdot[S_INDEX(YAW)] = this->params_.r / this->params_.L * (l_spd - r_spd);
}

__device__ void DiffDrive::computeStateDeriv(
    float* x, float* u,
    float* xdot, float* theta_s) {
  float l_spd = u[C_INDEX(LEFT_ROT_SPD)];
  float r_spd = u[C_INDEX(RIGHT_ROT_SPD)];
  float yaw = angle_utils::normalizeAngle(x[S_INDEX(YAW)]);
  // int tdy = threadIdx.y;
  // switch (tdy)
  // {
  //   case S_INDEX(X):
      xdot[S_INDEX(X)] = this->params_.r / 2.0f * (l_spd + r_spd) * __cosf(yaw);
      // break;
    // case S_INDEX(Y):
      xdot[S_INDEX(Y)] = this->params_.r / 2.0f * (l_spd + r_spd) * __sinf(yaw);
      // break;
    // case S_INDEX(YAW):
      xdot[S_INDEX(YAW)]   = this->params_.r / this->params_.L * (l_spd - r_spd);
      // break;
  // }
}

DiffDrive::state_array DiffDrive::stateFromMap(const std::map<std::string, float>& map) {
  state_array s;
  s[S_INDEX(X)] = map.at("X");
  s[S_INDEX(Y)] = map.at("POS_Y");
  s[S_INDEX(YAW)] = map.at("YAW");
  return s;
}
