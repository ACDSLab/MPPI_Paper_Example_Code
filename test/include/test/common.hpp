#include <chrono>
#include <ctime>
#include <fstream>

#include <stdlib.h>

struct CommonSettings
{
  static const int num_timesteps = 100;
  const int iteration_count = 1;
  const float resolution = 0.1;       // [meter / cell]
  const float footprint_size = 0.15;  // [m]
  const float origin_x = 0.0f;        // [m]
  const float origin_y = 0.0f;        // [m]
  const float length_x = 11.0f;       // [m]
  const float length_y = 11.0f;       // [m]
  const float lookahead_dist = 10.0;  // [m]
  static const int NUM_OBSTACLES = 3;
  float obstacle_pos_x[NUM_OBSTACLES] = { 8.0, 4.0, 6.0 };  // [m]
  float obstacle_pos_y[NUM_OBSTACLES] = { 8.0, 6.0, 5.0 };  // [m]
  float obstacle_sizes[NUM_OBSTACLES] = { 0.4, 0.5, 0.3 };  // [m]
  const unsigned char costmap_default_val = 0;
  const unsigned char obstacle_cost = 250;

  const float start_x = 1.0f;    // [m]
  const float start_y = 1.0f;    // [m]
  const float start_vel = 0.0f;  // [m/s]

  const float goal_x = 10.0f;   // [m]
  const float goal_y = 10.0f;   // [m]
  const float goal_yaw = 0.0f;  // [rad]

  const int num_iterations = 10000;

  const float dt = 0.02f;
  const float lambda = 1.0f;
  const float goal_weight = 5.0f;
  const int goal_power = 1;
  const float goal_angle_weight = 5.0f;
  const int goal_angle_power = 1;
  const float goal_dist_threshold = 1000.0f;    // [m]
  const float inscribed_radius = 0.0f;          // [m]
  const float circumscribed_radius = 0.226274;  // [m]

  const bool consider_footprint = false;
  const float near_goal_distance = 0.5f;    // [m]
  const float obs_inflation_radius = 0.1f;  // [m]
  const float obs_repulsion_weight = 0.0f;
  const float obs_scaling_factor = 1.0f;
  const float obs_traj_weight = 20.0f;
  const int obs_power = 1;

  const float std_dev_v = 0.2f;      // [m/s]
  const float std_dev_omega = 0.2f;  // [rad]
  const float robot_radius = 1.0f;   // [m]
  const float robot_length = 1.0f;   // [m]
  const float v_min = -0.35f;        // [m/s]
  const float v_max = 0.5f;          // [m/s]
  const float omega_max = 0.5f;      // [rad/s]
};

struct AutorallySettings
{
  static const int num_timesteps = 96;
  static const int DYN_BLOCK_X = 32;
  static const int DYN_BLOCK_Y = 16;
  const int COST_BLOCK_X = 96;

  const float dt = 1.0f/50.0f;
  const int optimization_stride = 1;
  const int num_iterations = 10000;
  const int num_iters = 1;

  const float lambda = 6.66f;
  const float alpha = 0.0f;
  const float discount = 0.9f;

  const float init_throttle = 0.0f;
  const float init_steering = 0.0f;
  const float std_throttle = 0.275f;
  const float std_steering = 0.3f;

  const float2 throttle_range = make_float2(-0.99f, 0.65f);
  const float2 steering_range = make_float2(-0.99f, 0.65f);

  // Cost setttings
  const float speed_coeff = 4.25f;
  const float track_coeff = 200.0f;
  const float slip_coeff = 10.0f;
  const float crash_coeff = 10000.0f;
  const float steering_coeff = 1.0f;
  const float throttle_coeff = 1.0f;

  const float desired_speed = 6.0f; // [m/s]
  const float max_slip_angle = 0.9f; // [rad]
  const float boundary_threshold = 0.65f;
  const float track_slop = 0.0f;
};

template <typename T = double>
class RunningStats
{
public:
  void add(const T& val)
  {
    count++;
    if (count == 1)
    {
      mean_ms_ = val;
    }
    else
    {
      T new_mean = mean_ms_ + (val - mean_ms_) / count;
      T new_variance = variance_ms_ + (val - mean_ms_) * (val - new_mean);
      mean_ms_ = new_mean;
      variance_ms_ = new_variance;
    }
  }

  T mean() const
  {
    return (count > 0) ? mean_ms_ : 0.0;
  }

  T variance() const
  {
    return (count > 1) ? variance_ms_ / (count - 1) : 0.0;
  }

protected:
  T mean_ms_ = 0.0;
  T variance_ms_ = 0.0;
  size_t count = 0;
};

inline std::string getCPUModelName()
{
  std::string cpu_name;
  FILE* cpuinfo = fopen("/proc/cpuinfo", "rb");
  if (cpuinfo == NULL)
    exit(EXIT_FAILURE);
  char* line = NULL;
  size_t len = 0;
  std::string search_string = "model name";
  while (getline(&line, &len, cpuinfo) != -1)
  {
    cpu_name = std::string(line);
    std::size_t found = cpu_name.find(search_string);
    if (found != std::string::npos)
    {
      found = cpu_name.find(": ");
      cpu_name = cpu_name.substr(found + 2, cpu_name.size() - 3 - found);
      break;
    }
  }

  free(line);
  fclose(cpuinfo);
  return cpu_name;
}

inline std::string getTimestamp()
{
  // Create timestamp
  std::time_t t = std::time(nullptr);
  const int buf_size = 100;
  char time_buf[buf_size];
  std::strftime(time_buf, buf_size, "%F_%H-%M-%S", std::localtime(&t));
  return std::string(time_buf);
}

static const std::string default_csv_header =
    "Processor,GPU,Method,Num Rollouts,Mean Optimization Time (ms), Std. Dev. "
    "Time (ms)\n";

inline void createNewCSVFile(std::string prefix, std::ofstream& new_file, std::string header = default_csv_header)
{
  new_file.open(prefix + "_" + getTimestamp() + ".csv");
  new_file << header;
}
