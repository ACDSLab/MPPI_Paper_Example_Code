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

class ROS2MPPITest : public ::testing::TestWithParam<int>
{
protected:
  // Settings
  std::string node_name = "MPPI";
  std::string motion_model = "DiffDrive";
  std::vector<std::string> critics = { "GoalCritic", "GoalAngleCritic", "ObstaclesCritic" };
  CommonSettings settings;
  double controller_frequency = 50.0;

  // ROS Variables
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros;
  std::vector<rclcpp::Parameter> params;
  rclcpp::NodeOptions options;
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node;
  std::shared_ptr<mppi::Optimizer> optimizer;
  std_msgs::msg::Header header;

  geometry_msgs::msg::PoseStamped start_pose;
  geometry_msgs::msg::Twist start_velocity;
  nav_msgs::msg::Path goals;
  nav2_core::GoalChecker* goal_checker{nullptr};

  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    rcutils_logging_set_logger_level("costmap", RCUTILS_LOG_SEVERITY_FATAL);
    rcutils_logging_set_logger_level(node_name.c_str(), RCUTILS_LOG_SEVERITY_FATAL);

    /**
     * Create Costmap
     **/
    const unsigned int cells_x = settings.length_x / settings.resolution;
    const unsigned int cells_y = settings.length_y / settings.resolution;
    costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("costmap");
    costmap_ros->on_configure(rclcpp_lifecycle::State{});
    auto costmap = std::make_shared<nav2_costmap_2d::Costmap2D>(cells_x, cells_y,
        settings.resolution, settings.origin_x, settings.origin_y, settings.costmap_default_val);
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
    // params.emplace_back(rclcpp::Parameter(node_name + ".ObstaclesCritic.enabled", false));
    params.emplace_back(rclcpp::Parameter("controller_frequency", controller_frequency));
    options.parameter_overrides(params);

    node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(node_name, options);
    auto parameters_handler = std::make_unique<mppi::ParametersHandler>(node);
    optimizer = std::make_shared<mppi::Optimizer>();
    std::weak_ptr<rclcpp_lifecycle::LifecycleNode> weak_ptr_node{node};
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

    auto goal_pose = start_pose;
    goal_pose.pose.position.x = settings.goal_x;
    goal_pose.pose.position.y = settings.goal_y;
    goals.poses.push_back(goal_pose);
    goals.header = header;
  }
  void TearDown() override
  {
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
    optimizer->evalControl(start_pose, start_velocity, goals, goal_checker);
    auto end = std::chrono::steady_clock::now();
    double duration = (end - start).count() / 1e6;
    times.add(duration);
  }
  printf("ROS2 MPPI with %d rollouts optimization time: %f +- %f ms\n",
         GetParam(),
         times.mean(), sqrt(times.variance()));
}

INSTANTIATE_TEST_CASE_P(ROS2MPPIComparisons,
    ROS2MPPITest,
    ::testing::Values(128, 256, 512, 1024, 2048, 4096, 6144, 8192, 4096 * 4));
