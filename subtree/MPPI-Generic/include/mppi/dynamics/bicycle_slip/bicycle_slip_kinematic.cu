//
// Created by jason on 12/12/22.
//

#include "bicycle_slip_kinematic.cuh"

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::BicycleSlipKinematicImpl(cudaStream_t stream)
  : PARENT_CLASS(stream)
{
  this->requires_buffer_ = true;
  steer_lstm_lstm_helper_ = std::make_shared<STEER_NN>(stream);
  delay_lstm_lstm_helper_ = std::make_shared<DELAY_NN>(stream);
  terra_lstm_lstm_helper_ = std::make_shared<TERRA_NN>(stream);
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::BicycleSlipKinematicImpl(const std::string& model_path,
                                                                                       cudaStream_t stream)
  : BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::BicycleSlipKinematicImpl(stream)
{
  if (!fileExists(model_path))
  {
    std::cerr << "Could not load neural net model at model_path: " << model_path.c_str();
    exit(-1);
  }
  cnpy::npz_t param_dict = cnpy::npz_load(model_path);

  // load the delay params
  this->params_.brake_delay_constant = param_dict.at("delay_model/params/constant").data<float>()[0];
  this->params_.max_brake_rate_neg = param_dict.at("delay_model/params/max_rate_neg").data<float>()[0];
  this->params_.max_brake_rate_pos = param_dict.at("delay_model/params/max_rate_pos").data<float>()[0];

  // load the steering parameters
  this->params_.max_steer_rate = param_dict.at("steer_model/params/max_rate_pos").data<float>()[0];
  this->params_.steering_constant = param_dict.at("steer_model/params/constant").data<float>()[0];

  delay_lstm_lstm_helper_->loadParams("delay_model/model", model_path);
  steer_lstm_lstm_helper_->loadParams("steer_model/model", model_path);
  terra_lstm_lstm_helper_->loadParams("bicycle_model", model_path);
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::state_array
BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::stateFromMap(const std::map<std::string, float>& map)
{
  state_array s = state_array::Zero();
  if (map.find("VEL_X") == map.end() || map.find("VEL_Y") == map.end() || map.find("POS_X") == map.end() ||
      map.find("POS_Y") == map.end() || map.find("ROLL") == map.end() || map.find("PITCH") == map.end())
  {
    std::cout << "WARNING: could not find all keys for ackerman slip dynamics" << std::endl;
    for (const auto& it : map)
    {
      std::cout << "got key " << it.first << std::endl;
    }
    return s;
  }
  s(S_INDEX(POS_X)) = map.at("POS_X");
  s(S_INDEX(POS_Y)) = map.at("POS_Y");
  s(S_INDEX(VEL_X)) = map.at("VEL_X");
  s(S_INDEX(VEL_Y)) = map.at("VEL_Y");
  s(S_INDEX(OMEGA_Z)) = map.at("OMEGA_Z");
  s(S_INDEX(YAW)) = map.at("YAW");
  s(S_INDEX(ROLL)) = map.at("ROLL");
  s(S_INDEX(PITCH)) = map.at("PITCH");
  if (map.find("STEER_ANGLE") != map.end())
  {
    s(S_INDEX(STEER_ANGLE)) = map.at("STEER_ANGLE");
    s(S_INDEX(STEER_ANGLE_RATE)) = map.at("STEER_ANGLE_RATE");
  }
  else
  {
    std::cout << "WARNING: unable to find BRAKE_STATE or STEER_ANGLE_RATE, using 0" << std::endl;
    s(S_INDEX(STEER_ANGLE)) = 0;
    s(S_INDEX(STEER_ANGLE_RATE)) = 0;
  }
  if (map.find("BRAKE_STATE") != map.end())
  {
    s(S_INDEX(BRAKE_STATE)) = map.at("BRAKE_STATE");
  }
  else if (map.find("BRAKE_CMD") != map.end())
  {
    std::cout << "WARNING: unable to find BRAKE_STATE" << std::endl;
    s(S_INDEX(BRAKE_STATE)) = map.at("BRAKE_CMD");
  }
  else
  {
    std::cout << "WARNING: unable to find BRAKE_CMD or BRAKE_STATE" << std::endl;
    s(S_INDEX(BRAKE_STATE)) = 0;
  }
  return s;
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::GPUSetup()
{
  steer_lstm_lstm_helper_->GPUSetup();
  delay_lstm_lstm_helper_->GPUSetup();
  terra_lstm_lstm_helper_->GPUSetup();

  // makes sure that the device ptr sees the correct lstm model
  this->steer_network_d_ = steer_lstm_lstm_helper_->getLSTMDevicePtr();
  this->delay_network_d_ = delay_lstm_lstm_helper_->getLSTMDevicePtr();
  this->terra_network_d_ = terra_lstm_lstm_helper_->getLSTMDevicePtr();

  PARENT_CLASS::GPUSetup();
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::freeCudaMem()
{
  steer_lstm_lstm_helper_->freeCudaMem();
  delay_lstm_lstm_helper_->freeCudaMem();
  terra_lstm_lstm_helper_->freeCudaMem();
  PARENT_CLASS::freeCudaMem();
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::updateFromBuffer(
    const typename PARENT_CLASS::buffer_trajectory& buffer)
{
  if (buffer.find("VEL_X") == buffer.end() || buffer.find("VEL_Y") == buffer.end() ||
      buffer.find("STEER_ANGLE") == buffer.end() || buffer.find("STEER_ANGLE_RATE") == buffer.end() ||
      buffer.find("STEER_CMD") == buffer.end() || buffer.find("BRAKE_STATE") == buffer.end())
  {
    std::cout << "WARNING: not using init buffer" << std::endl;
    for (const auto& it : buffer)
    {
      std::cout << "got key " << it.first << std::endl;
    }
    return;
  }

  DELAY_NN::init_buffer delay_init_buffer;
  delay_init_buffer.row(0) = buffer.at("BRAKE_STATE");
  delay_init_buffer.row(1) = buffer.at("BRAKE_CMD");
  delay_lstm_lstm_helper_->initializeLSTM(delay_init_buffer);

  STEER_NN::init_buffer steer_init_buffer;
  steer_init_buffer.row(0) = buffer.at("VEL_X") / 20.0f;
  steer_init_buffer.row(1) = buffer.at("STEER_ANGLE") / 5.0f;
  steer_init_buffer.row(2) = buffer.at("STEER_ANGLE_RATE") / 10.0f;
  steer_init_buffer.row(3) = buffer.at("STEER_CMD");
  steer_lstm_lstm_helper_->initializeLSTM(steer_init_buffer);

  typename TERRA_NN::init_buffer terra_init_buffer = TERRA_NN::init_buffer::Ones();
  terra_init_buffer.row(0) = buffer.at("VEL_X") / 20.0f;
  terra_init_buffer.row(1) = buffer.at("VEL_Y") / 5.0f;
  terra_init_buffer.row(2) = buffer.at("OMEGA_Z") / 5.0f;
  terra_init_buffer.row(3) = buffer.at("THROTTLE_CMD");
  terra_init_buffer.row(4) = buffer.at("BRAKE_STATE");
  terra_init_buffer.row(5) = buffer.at("STEER_ANGLE") / 5.0f;
  // TODO should be pulled from elevation map to be entirely correct
  terra_init_buffer.row(6) = buffer.at("PITCH");
  terra_init_buffer.row(7) = buffer.at("ROLL");
  terra_init_buffer.row(8) *= this->params_.environment;

  terra_lstm_lstm_helper_->initializeLSTM(terra_init_buffer);
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::initializeDynamics(
    const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
    Eigen::Ref<output_array> output, float t_0, float dt)
{
  this->steer_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  this->delay_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  this->terra_lstm_lstm_helper_->resetLSTMHiddenCellCPU();
  PARENT_CLASS::initializeDynamics(state, control, output, t_0, dt);
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::computeDynamics(
    const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
    Eigen::Ref<state_array> state_der)
{
  state_der = state_array::Zero();
  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0;
  float brake_cmd = -enable_brake * control(C_INDEX(THROTTLE_BRAKE));
  float throttle_cmd = !enable_brake * control(C_INDEX(THROTTLE_BRAKE));

  // runs parametric delay model
  this->computeParametricDelayDeriv(state, control, state_der);

  if (this->params_.enable_delay_model)
  {
    // runs the brake model
    DELAY_LSTM::input_array brake_input;
    brake_input(0) = state(S_INDEX(BRAKE_STATE));
    brake_input(1) = brake_cmd;
    brake_input(2) = state_der(S_INDEX(BRAKE_STATE));  // stand in for y velocity
    DELAY_LSTM::output_array brake_output = DELAY_LSTM::output_array::Zero();
    delay_lstm_lstm_helper_->forward(brake_input, brake_output);
    state_der(S_INDEX(BRAKE_STATE)) += brake_output(0);
  }

  // runs the parametric part of the steering model
  this->computeParametricSteerDeriv(state, control, state_der);

  // runs the steering model
  STEER_LSTM::input_array steer_input;
  steer_input(0) = state(S_INDEX(VEL_X)) / 20.0f;
  steer_input(1) = state(S_INDEX(STEER_ANGLE)) / 5.0f;
  steer_input(2) = state(S_INDEX(STEER_ANGLE_RATE)) / 10.0f;
  steer_input(3) = control(C_INDEX(STEER_CMD));
  steer_input(4) = state_der(S_INDEX(STEER_ANGLE)) / 10.0f;  // this is the parametric part as input
  STEER_LSTM::output_array steer_output = STEER_LSTM::output_array::Zero();
  steer_lstm_lstm_helper_->forward(steer_input, steer_output);
  state_der(S_INDEX(STEER_ANGLE)) += steer_output(0) * 10;

  // if in reverse use parametric model
  if (this->params_.gear_sign == -1)
  {
    // TODO dt is not known here
    this->computeParametricAccelDeriv(state, control, state_der, 0.02);

    // kinematic
    state_der(S_INDEX(OMEGA_Z)) = 0.0f;
    state_der(S_INDEX(VEL_Y)) = 0.0f;
  }
  else
  {
    // runs the terra dynamics model
    typename TERRA_LSTM::input_array terra_input;
    terra_input(0) = state(S_INDEX(VEL_X)) / 20.0f;
    terra_input(1) = state(S_INDEX(VEL_Y)) / 5.0f;
    terra_input(2) = state(S_INDEX(OMEGA_Z)) / 5.0f;
    terra_input(3) = throttle_cmd;
    terra_input(4) = state(S_INDEX(BRAKE_STATE));
    terra_input(5) = state(S_INDEX(STEER_ANGLE)) / 5.0f;
    // if roll/pitch is invalid just set it to zero
    terra_input(6) = state(S_INDEX(PITCH)) * (abs(state(S_INDEX(PITCH))) < M_PI_2f32);
    terra_input(7) = state(S_INDEX(ROLL)) * (abs(state(S_INDEX(ROLL))) < M_PI_2f32);
    terra_input(8) = this->params_.environment;
    typename TERRA_LSTM::output_array terra_output = TERRA_LSTM::output_array::Zero();
    terra_lstm_lstm_helper_->forward(terra_input, terra_output);

    // combine to compute state derivative
    state_der(S_INDEX(VEL_X)) = terra_output(0) * 10.0f;
    state_der(S_INDEX(VEL_Y)) = terra_output(1) * 5.0f;
    state_der(S_INDEX(OMEGA_Z)) = terra_output(2) * 5.0f;

    // kinematic component
    state_der(S_INDEX(YAW)) = state(S_INDEX(OMEGA_Z));
  }

  // kinematics component
  state_der(S_INDEX(POS_X)) =
      state(S_INDEX(VEL_X)) * cosf(state(S_INDEX(YAW))) - state(S_INDEX(VEL_Y)) * sinf(state(S_INDEX(YAW)));
  state_der(S_INDEX(POS_Y)) =
      state(S_INDEX(VEL_X)) * sinf(state(S_INDEX(YAW))) + state(S_INDEX(VEL_Y)) * cosf(state(S_INDEX(YAW)));
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::updateState(
    const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
    const float dt)
{
  next_state = state + state_der * dt;
  if (this->params_.gear_sign == -1)
  {
    next_state(S_INDEX(OMEGA_Z)) = state_der(S_INDEX(YAW));
    next_state(S_INDEX(VEL_Y)) = 0.0f;
  }
  next_state(S_INDEX(YAW)) = angle_utils::normalizeAngle(next_state(S_INDEX(YAW)));
  next_state(S_INDEX(STEER_ANGLE)) =
      max(min(next_state(S_INDEX(STEER_ANGLE)), this->params_.max_steer_angle), -this->params_.max_steer_angle);
  next_state(S_INDEX(STEER_ANGLE_RATE)) = state_der(S_INDEX(STEER_ANGLE));
  next_state(S_INDEX(BRAKE_STATE)) =
      min(max(next_state(S_INDEX(BRAKE_STATE)), 0.0f), -this->control_rngs_[C_INDEX(THROTTLE_BRAKE)].x);
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::step(
    Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
    const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output, const float t, const float dt)
{
  CLASS_T* derived = static_cast<CLASS_T*>(this);
  derived->computeDynamics(state, control, state_der);
  derived->updateState(state, next_state, state_der, dt);

  output = output_array::Zero();

  float roll = state(S_INDEX(ROLL));
  float pitch = state(S_INDEX(PITCH));
  RACER::computeStaticSettling<typename DYN_PARAMS_T::OutputIndex, TwoDTextureHelper<float>>(
      this->tex_helper_, next_state(S_INDEX(YAW)), next_state(S_INDEX(POS_X)), next_state(S_INDEX(POS_Y)), roll, pitch,
      output.data());
  next_state[S_INDEX(PITCH)] = pitch;
  next_state[S_INDEX(ROLL)] = roll;

  this->setOutputs(state_der.data(), next_state.data(), output.data());

  output[O_INDEX(BASELINK_VEL_B_Y)] = next_state[S_INDEX(VEL_Y)];
  output[O_INDEX(ACCEL_Y)] = state_der[S_INDEX(VEL_Y)];
  output[O_INDEX(OMEGA_Z)] = next_state[S_INDEX(OMEGA_Z)];
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
__device__ void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::initializeDynamics(
    float* state, float* control, float* output, float* theta_s, float t_0, float dt)
{
  const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES / 4 + 1;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {  // Allows us to turn on or off global or shared memory version of params
    DYN_PARAMS_T* dyn_params = (DYN_PARAMS_T*)theta_s;
    *dyn_params = this->params_;
  }
  SHARED_MEM_GRD_PARAMS* shared_params = (SHARED_MEM_GRD_PARAMS*)(theta_s + shift);

  // setup memory for hidden/cell state memory
  //
  // if we are using shared memory load in the parameters
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    SHARED_MEM_BLK_PARAMS* blk_params = (SHARED_MEM_BLK_PARAMS*)(shared_params + 1);
    blk_params += blockDim.x * threadIdx.z + threadIdx.x;
    steer_network_d_->initialize(&shared_params->steer_lstm_params, &shared_params->steer_output_params,
                                 &blk_params->steer_hidden_cell[0]);
    delay_network_d_->initialize(&shared_params->delay_lstm_params, &shared_params->delay_output_params,
                                 &blk_params->delay_hidden_cell[0]);
    terra_network_d_->initialize(&shared_params->terra_lstm_params, &shared_params->terra_output_params,
                                 &blk_params->terra_hidden_cell[0]);
  }
  else
  {
    SHARED_MEM_BLK_PARAMS* blk_params = (SHARED_MEM_BLK_PARAMS*)(shared_params);
    blk_params += blockDim.x * threadIdx.z + threadIdx.x;
    // only setup the hidden/cell states
    steer_network_d_->initialize(nullptr, nullptr, &blk_params->steer_hidden_cell[0]);
    delay_network_d_->initialize(nullptr, nullptr, &blk_params->delay_hidden_cell[0]);
    terra_network_d_->initialize(nullptr, nullptr, &blk_params->terra_hidden_cell[0]);
    __syncthreads();
  }
  for (int i = 0; i < this->OUTPUT_DIM && i < this->STATE_DIM; i++)
  {
    output[i] = state[i];
  }
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
__device__ void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::updateState(
    float* state, float* next_state, float* state_der, const float dt, typename PARENT_CLASS::DYN_PARAMS_T* params_p)
{
  for (int i = threadIdx.y; i < 8; i += blockDim.y)
  {
    next_state[i] = state[i] + state_der[i] * dt;
    switch (i)
    {
      case S_INDEX(YAW):
        next_state[i] = angle_utils::normalizeAngle(next_state[i]);
        break;
      case S_INDEX(STEER_ANGLE):
        next_state[S_INDEX(STEER_ANGLE)] =
            max(min(next_state[S_INDEX(STEER_ANGLE)], params_p->max_steer_angle), -params_p->max_steer_angle);
        next_state[S_INDEX(STEER_ANGLE_RATE)] = state_der[S_INDEX(STEER_ANGLE)];
        break;
      case S_INDEX(BRAKE_STATE):
        next_state[S_INDEX(BRAKE_STATE)] =
            min(max(next_state[S_INDEX(BRAKE_STATE)], 0.0f), -this->control_rngs_[C_INDEX(THROTTLE_BRAKE)].x);
    }
  }

  __syncthreads();

  if (params_p->gear_sign == -1)
  {
    next_state[S_INDEX(OMEGA_Z)] = state_der[S_INDEX(YAW)];
    next_state[S_INDEX(VEL_Y)] = 0.0f;
  }
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
__device__ void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::computeDynamics(float* state,
                                                                                              float* control,
                                                                                              float* state_der,
                                                                                              float* theta)
{
  DYN_PARAMS_T* params_p = nullptr;

  const int shift = PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES / 4 + 1;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {  // Allows us to turn on or off global or shared memory version of params
    params_p = (DYN_PARAMS_T*)theta;
  }
  else
  {
    params_p = &(this->params_);
  }

  // nullptr if not shared memory
  SHARED_MEM_GRD_PARAMS* params = (SHARED_MEM_GRD_PARAMS*)(theta + shift);
  SHARED_MEM_BLK_PARAMS* blk_params = (SHARED_MEM_BLK_PARAMS*)params;
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    // if GRD in shared them
    blk_params = (SHARED_MEM_BLK_PARAMS*)(params + 1);
  }
  blk_params = blk_params + blockDim.x * threadIdx.z + threadIdx.x;
  float* theta_s_shifted = &blk_params->theta_s[0];

  bool enable_brake = control[C_INDEX(THROTTLE_BRAKE)] < 0;
  const float brake_cmd = -enable_brake * control[C_INDEX(THROTTLE_BRAKE)];
  const float throttle_cmd = !enable_brake * control[C_INDEX(THROTTLE_BRAKE)];

  // parametric part of the brake
  this->computeParametricDelayDeriv(state, control, state_der, params_p);

  // runs the parametric part of the steering model
  this->computeParametricSteerDeriv(state, control, state_der, params_p);

  // runs the brake model
  float* input_loc = &theta_s_shifted[DELAY_LSTM::HIDDEN_DIM];
  float* output = nullptr;

  if (params_p->enable_delay_model)
  {
    input_loc[0] = state[S_INDEX(BRAKE_STATE)];
    input_loc[1] = brake_cmd;
    input_loc[2] = state_der[S_INDEX(BRAKE_STATE)];  // stand in for y velocity

    if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
    {
      output = delay_network_d_->forward(nullptr, theta_s_shifted, &blk_params->delay_hidden_cell[0],
                                         &params->delay_lstm_params, &params->delay_output_params, 0);
    }
    else
    {
      output =
          delay_network_d_->forward(nullptr, theta_s_shifted, &blk_params->delay_hidden_cell[0],
                                    &delay_network_d_->params_, delay_network_d_->getOutputModel()->getParamsPtr(), 0);
    }
    if (threadIdx.y == 0)
    {
      state_der[S_INDEX(BRAKE_STATE)] += output[0];
    }
  }

  // runs the steering model
  __syncthreads();  // required since we can overwrite the output before grabbing it
  input_loc = &theta_s_shifted[STEER_LSTM::HIDDEN_DIM];
  input_loc[0] = state[S_INDEX(VEL_X)] / 20.0f;
  input_loc[1] = state[S_INDEX(STEER_ANGLE)] / 5.0f;
  input_loc[2] = state[S_INDEX(STEER_ANGLE_RATE)] / 10.0f;
  input_loc[3] = control[C_INDEX(STEER_CMD)];
  input_loc[4] = state_der[S_INDEX(STEER_ANGLE)] / 10.0f;  // this is the parametric part as input
  if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {
    output = steer_network_d_->forward(nullptr, theta_s_shifted, &blk_params->steer_hidden_cell[0],
                                       &params->steer_lstm_params, &params->steer_output_params, 0);
  }
  else
  {
    output =
        steer_network_d_->forward(nullptr, theta_s_shifted, &blk_params->steer_hidden_cell[0],
                                  &steer_network_d_->params_, steer_network_d_->getOutputModel()->getParamsPtr(), 0);
  }
  if (threadIdx.y == 0)
  {
    state_der[S_INDEX(STEER_ANGLE)] += output[0] * 10.0f;
  }
  __syncthreads();  // required since we can overwrite the output before grabbing it

  if (params_p->gear_sign == -1)
  {
    // in reverse
    this->computeParametricAccelDeriv(state, control, state_der, 0.02, params_p);

    // kinematic
    state_der[S_INDEX(OMEGA_Z)] = 0.0f;
    state_der[S_INDEX(VEL_Y)] = 0.0f;
  }
  else
  {
    // runs the terra dynamics model
    input_loc = &theta_s_shifted[TERRA_LSTM::HIDDEN_DIM];
    input_loc[0] = state[S_INDEX(VEL_X)] / 20.0f;
    input_loc[1] = state[S_INDEX(VEL_Y)] / 5.0f;
    input_loc[2] = state[S_INDEX(OMEGA_Z)] / 5.0f;
    input_loc[3] = throttle_cmd;
    input_loc[4] = state[S_INDEX(BRAKE_STATE)];
    input_loc[5] = state[S_INDEX(STEER_ANGLE)] / 5.0f;
    input_loc[6] = state[S_INDEX(PITCH)] * (abs(state[S_INDEX(PITCH)]) < M_PI_2f32);
    input_loc[7] = state[S_INDEX(ROLL)] * (abs(state[S_INDEX(ROLL)]) < M_PI_2f32);
    input_loc[8] = this->params_.environment;

    if (SHARED_MEM_REQUEST_GRD_BYTES != 0)
    {
      output = terra_network_d_->forward(nullptr, theta_s_shifted, &blk_params->terra_hidden_cell[0],
                                         &params->terra_lstm_params, &params->terra_output_params, 0);
    }
    else
    {
      output =
          terra_network_d_->forward(nullptr, theta_s_shifted, &blk_params->terra_hidden_cell[0],
                                    &terra_network_d_->params_, terra_network_d_->getOutputModel()->getParamsPtr(), 0);
    }

    // combine to compute state derivative
    state_der[S_INDEX(VEL_X)] = output[0] * 10.0f;
    state_der[S_INDEX(VEL_Y)] = output[1] * 5.0f;
    state_der[S_INDEX(OMEGA_Z)] = output[2] * 5.0f;

    state_der[S_INDEX(YAW)] = state[S_INDEX(OMEGA_Z)];
  }

  // kinematics component
  state_der[S_INDEX(POS_X)] =
      state[S_INDEX(VEL_X)] * cosf(state[S_INDEX(YAW)]) - state[S_INDEX(VEL_Y)] * sinf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_Y)] =
      state[S_INDEX(VEL_X)] * sinf(state[S_INDEX(YAW)]) + state[S_INDEX(VEL_Y)] * cosf(state[S_INDEX(YAW)]);
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
__device__ void BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::step(float* state, float* next_state,
                                                                                   float* state_der, float* control,
                                                                                   float* output, float* theta_s,
                                                                                   const float t, const float dt)
{
  DYN_PARAMS_T* params_p;
  if (PARENT_CLASS::SHARED_MEM_REQUEST_GRD_BYTES != 0)
  {  // Allows us to turn on or off global or shared memory version of params
    params_p = (DYN_PARAMS_T*)theta_s;
  }
  else
  {
    params_p = &(this->params_);
  }
  const uint tdy = threadIdx.y;

  CLASS_T* derived = static_cast<CLASS_T*>(this);
  derived->computeDynamics(state, control, state_der, theta_s);
  derived->updateState(state, next_state, state_der, dt, params_p);

  if (tdy == 0)
  {
    float roll = state[S_INDEX(ROLL)];
    float pitch = state[S_INDEX(PITCH)];
    RACER::computeStaticSettling<DYN_PARAMS_T::OutputIndex, TwoDTextureHelper<float>>(
        this->tex_helper_, next_state[S_INDEX(YAW)], next_state[S_INDEX(POS_X)], next_state[S_INDEX(POS_Y)], roll,
        pitch, output);
    next_state[S_INDEX(PITCH)] = pitch;
    next_state[S_INDEX(ROLL)] = roll;

    this->setOutputs(state_der, next_state, output);

    output[O_INDEX(BASELINK_VEL_B_Y)] = next_state[S_INDEX(VEL_Y)];
    output[O_INDEX(ACCEL_Y)] = state_der[S_INDEX(VEL_Y)];
    output[O_INDEX(OMEGA_Z)] = next_state[S_INDEX(OMEGA_Z)];
  }
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
Eigen::Vector3f BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::velocityFromState(
    const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(state[S_INDEX(VEL_X)], state(S_INDEX(VEL_Y)), 0);
}

template <class CLASS_T, class PARAMS_T, int TERRA_INPUT_DIM>
Eigen::Vector3f BicycleSlipKinematicImpl<CLASS_T, PARAMS_T, TERRA_INPUT_DIM>::angularRateFromState(
    const Eigen::Ref<const state_array>& state)
{
  return Eigen::Vector3f(0, 0, state[S_INDEX(OMEGA_Z)]);
}