#include <gtest/gtest.h>
#include <nav2_costmap_2d/inflation_layer.hpp>
#include <nav2_mppi_controller/optimizer.hpp>
#include <rclcpp/rclcpp.hpp>

#include "test/common.hpp"

void addObstacle(nav2_costmap_2d::Costmap2D* costmap, unsigned int upper_left_corner_x,
                 unsigned int upper_left_corner_y, unsigned int size, unsigned char cost)
{
  for (unsigned int i = upper_left_corner_x; i < upper_left_corner_x + size; i++)
  {
    for (unsigned int j = upper_left_corner_y; j < upper_left_corner_y + size; j++)
    {
      costmap->setCost(i, j, cost);
    }
  }
}

class CSVWritingEnvironment : public ::testing::Environment
{
public:
  static std::ofstream csv_file;
  static std::string cpu_name;
  static std::string gpu_name;
  ~CSVWritingEnvironment() override {}
  void SetUp() override {
    createNewCSVFile("ros2_results", csv_file);

    // get CPU name
    cpu_name = getCPUModelName();
  }

  void TearDown() override{
    csv_file.close();
  }

  static std::string getGPUName()
  {
    return gpu_name;
  }

  static std::string getCPUName()
  {
    return cpu_name;
  }
};

// Iniitialize static variables
std::ofstream CSVWritingEnvironment::csv_file;
std::string CSVWritingEnvironment::cpu_name = "N/A";
std::string CSVWritingEnvironment::gpu_name = "N/A";

// Register Environment
testing::Environment* const csv_env = testing::AddGlobalTestEnvironment(new CSVWritingEnvironment);

class ROS2MPPITest : public ::testing::TestWithParam<int>
{
protected:
  // Settings
  std::string node_name = "MPPI";
  std::string motion_model = "DiffDrive";
  std::vector<std::string> critics = { "GoalCritic", "GoalAngleCritic", "ObstaclesCritic" };
  CommonSettings settings;
  double controller_frequency = 1.0 / settings.dt;

  // ROS Variables
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros;
  std::vector<rclcpp::Parameter> params;
  rclcpp::NodeOptions options;
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node;
#ifdef CMAKE_ROS_IRON
  std::shared_ptr<mppi::Optimizer> optimizer;
#else
  mppi::Optimizer* optimizer;
#endif
  std_msgs::msg::Header header;

  geometry_msgs::msg::PoseStamped start_pose, goal_pose;
  geometry_msgs::msg::Twist start_velocity;
  nav_msgs::msg::Path goals;
  nav2_core::GoalChecker* goal_checker{ nullptr };

  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    auto check = rcutils_logging_set_logger_level("costmap", RCUTILS_LOG_SEVERITY_FATAL);
    if (check)
    {
      std::cout << "Something went wrong with logging: " << check << std::endl;
    }
    auto check2 = rcutils_logging_set_logger_level(node_name.c_str(), RCUTILS_LOG_SEVERITY_FATAL);
    if (check2)
    {
      std::cout << "Something went wrong with logging: " << check2 << std::endl;
    }

    /**
     * Create Costmap
     **/
    const unsigned int cells_x = settings.length_x / settings.resolution;
    const unsigned int cells_y = settings.length_y / settings.resolution;
    costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("costmap");
    costmap_ros->on_configure(rclcpp_lifecycle::State{});
    auto costmap = std::make_shared<nav2_costmap_2d::Costmap2D>(
        cells_x, cells_y, settings.resolution, settings.origin_x, settings.origin_y, settings.costmap_default_val);
    *(costmap_ros->getCostmap()) = *costmap;

    std::vector<geometry_msgs::msg::Point> footprint;
    geometry_msgs::msg::Point point;
    point.z = 0;
    point.x = settings.footprint_size;
    point.y = settings.footprint_size;
    footprint.push_back(point);
    point.x = -settings.footprint_size;
    point.y = -settings.footprint_size;
    footprint.push_back(point);
    point.x = settings.footprint_size;
    point.y = -settings.footprint_size;
    footprint.push_back(point);
    point.x = -settings.footprint_size;
    point.y = settings.footprint_size;
    footprint.push_back(point);

    costmap_ros->setRobotFootprint(footprint);

    // Add Obstacles
    for (int i = 0; i < settings.NUM_OBSTACLES; i++)
    {
      int obs_x = settings.obstacle_pos_x[i] / settings.resolution;
      int obs_y = settings.obstacle_pos_y[i] / settings.resolution;
      int obs_size = settings.obstacle_sizes[i] / settings.resolution;
      addObstacle(costmap_ros->getCostmap(), obs_x, obs_y, obs_size, settings.obstacle_cost);
    }

    /**
     * Setup Optimizer
     **/
    params.emplace_back(rclcpp::Parameter(node_name + ".iteration_count", settings.iteration_count));
    params.emplace_back(rclcpp::Parameter(node_name + ".batch_size", GetParam()));
    params.emplace_back(rclcpp::Parameter(node_name + ".time_steps", settings.num_timesteps));
    params.emplace_back(rclcpp::Parameter(node_name + ".lookahead_dist", settings.lookahead_dist));
    params.emplace_back(rclcpp::Parameter(node_name + ".motion_model", motion_model));
    params.emplace_back(rclcpp::Parameter(node_name + ".critics", critics));
    params.emplace_back(rclcpp::Parameter(node_name + ".temperature", settings.lambda));
    params.emplace_back(rclcpp::Parameter(node_name + ".model_dt", settings.dt));
    params.emplace_back(rclcpp::Parameter(node_name + ".vx_std", settings.std_dev_v));
    params.emplace_back(rclcpp::Parameter(node_name + ".vy_std", settings.std_dev_v));
    params.emplace_back(rclcpp::Parameter(node_name + ".wz_std", settings.std_dev_omega));
    params.emplace_back(rclcpp::Parameter(node_name + ".vx_max", settings.v_max));
    params.emplace_back(rclcpp::Parameter(node_name + ".vx_min", settings.v_min));
    params.emplace_back(rclcpp::Parameter(node_name + ".vy_max", settings.v_max));
    params.emplace_back(rclcpp::Parameter(node_name + ".wz_max", settings.omega_max));
    params.emplace_back(rclcpp::Parameter(node_name + ".GoalCritic.cost_weight", settings.goal_weight));
    params.emplace_back(rclcpp::Parameter(node_name + ".GoalCritic.cost_power", settings.goal_power));
    params.emplace_back(
        rclcpp::Parameter(node_name + ".GoalCritic.threshold_to_consider", settings.goal_dist_threshold));
    params.emplace_back(rclcpp::Parameter(node_name + ".GoalAngleCritic.cost_weight", settings.goal_angle_weight));
    params.emplace_back(rclcpp::Parameter(node_name + ".GoalAngleCritic.cost_power", settings.goal_angle_power));
    params.emplace_back(
        rclcpp::Parameter(node_name + ".GoalAngleCritic.threshold_to_consider", settings.goal_dist_threshold));
    params.emplace_back(rclcpp::Parameter(node_name + ".ObstaclesCritic.cost_power", settings.obs_power));
    params.emplace_back(
        rclcpp::Parameter(node_name + ".ObstaclesCritic.repulsion_weight", settings.obs_repulsion_weight));
    params.emplace_back(rclcpp::Parameter(node_name + ".ObstaclesCritic.critical_weight", settings.obs_traj_weight));
    params.emplace_back(
        rclcpp::Parameter(node_name + ".ObstaclesCritic.consider_footprint", settings.consider_footprint));
    params.emplace_back(
        rclcpp::Parameter(node_name + ".ObstaclesCritic.near_goal_distance", settings.near_goal_distance));
    params.emplace_back(
        rclcpp::Parameter(node_name + ".ObstaclesCritic.inflation_radius", settings.obs_inflation_radius));
    params.emplace_back(
        rclcpp::Parameter(node_name + ".ObstaclesCritic.cost_scaling_factor", settings.obs_scaling_factor));
    params.emplace_back(rclcpp::Parameter("controller_frequency", controller_frequency));
    options.parameter_overrides(params);

    node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(node_name, options);
#ifdef CMAKE_ROS_IRON
    auto parameters_handler = std::make_unique<mppi::ParametersHandler>(node);
    optimizer = std::make_shared<mppi::Optimizer>();
#else
    auto parameters_handler = std::make_unique<mppi::ParametersHandler>(node, node_name);
    optimizer = new mppi::Optimizer();
#endif
    std::weak_ptr<rclcpp_lifecycle::LifecycleNode> weak_ptr_node{ node };
    optimizer->initialize(weak_ptr_node, node->get_name(), costmap_ros, parameters_handler.get());

    /**
     * Set up initial state and goal
     */
    header.frame_id = "odom";
    header.stamp = node->get_clock()->now();
    start_pose.header = header;
    start_pose.pose.position.x = settings.start_x;
    start_pose.pose.position.y = settings.start_y;

    start_velocity.linear.x = settings.start_vel;

    goal_pose = start_pose;
    goal_pose.pose.position.x = settings.goal_x;
    goal_pose.pose.position.y = settings.goal_y;
    goals.poses.push_back(goal_pose);
    goals.header = header;
  }
  void TearDown() override
  {
#ifdef CMAKE_ROS_IRON
    // TODO: Check if issue is also in ROS 2 iron. It wasn't when I originally tested
#else
    // TODO: Explicitly don't delete the optimizer as it causes double free corruption
    // This is happening in ROS2 kilted and is potentially related to
    // https://github.com/ros-navigation/navigation2/issues/3767
    // as the system I am testing on is an AMD system and probably doesn't have the same
    // architecture options as the system that built the ROS 2 mppi package.
    // Specifically the double free occurs with some Eigen variables in the optimizer
    // which if the ros2 mppi shared library built Eigen with different compilaition flags
    // compared to this executable, you would get weird memory issues

    // delete optimizer;
#endif
    rclcpp::shutdown();
  }
};

TEST_P(ROS2MPPITest, DifferentNumSamples)
{
  RunningStats times;
  for (int i = 0; i < settings.num_iterations; i++)
  {
    auto start = std::chrono::steady_clock::now();
    // run optimizer
#ifdef CMAKE_ROS_IRON
    optimizer->evalControl(start_pose, start_velocity, goals, goal_checker);
#else
    optimizer->evalControl(start_pose, start_velocity, goals, goal_pose.pose, goal_checker);
#endif
    auto end = std::chrono::steady_clock::now();
    double duration = (end - start).count() / 1e6;
    times.add(duration);
  }
  // Save to CSV File
  CSVWritingEnvironment::csv_file << CSVWritingEnvironment::getCPUName()
      << "," << CSVWritingEnvironment::getGPUName() << ",ros2,"
      << GetParam() << "," << times.mean()
      << "," << sqrt(times.variance()) << std::endl;
  printf("ROS2 MPPI with %d rollouts optimization time: %f +- %f ms\n", GetParam(), times.mean(),
         sqrt(times.variance()));
  printf("\tAverage Optimization Hz: %f Hz\n", 1000.0 / times.mean());
}

INSTANTIATE_TEST_CASE_P(ROS2MPPIComparisons, ROS2MPPITest,
                        ::testing::Values(128, 256, 512, 1024, 2048, 4096, 6144, 8192, 4096 * 4));
