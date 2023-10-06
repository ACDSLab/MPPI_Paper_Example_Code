#include <nav2_costmap_2d/inflation_layer.hpp>
#include <nav2_mppi_controller/optimizer.hpp>
#include <rclcpp/rclcpp.hpp>
#include <stdio.h>

#include <chrono>

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

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  std::string motion_model = "DiffDrive";
  std::vector<std::string> critics;
  critics.push_back("GoalCritic");
  critics.push_back("GoalAngleCritic");
  critics.push_back("ObstaclesCritic");
  bool consider_footprint = true;

  int batch_size = 2048;
  int time_steps = 100;
  int iteration_count = 1;
  double lookahead_dist = 10.0;
  double controller_frequency = 50.0;

  // Set up Costmap
  const unsigned int cells_x = 200;
  const unsigned int cells_y = 200;
  const double origin_x = 0.0;
  const double origin_y = 0.0;
  const double resolution = 0.1;
  const unsigned char cost_map_default_value = 0;
  const double footprint_size = 0.15;

  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("cost_map_node");
  costmap_ros->on_configure(rclcpp_lifecycle::State{});
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2D>(cells_x, cells_y, resolution, origin_x, origin_y,
                                                              cost_map_default_value);
  *(costmap_ros->getCostmap()) = *costmap;
  std::vector<geometry_msgs::msg::Point> footprint;
  geometry_msgs::msg::Point point;
  point.z = 0;
  point.x = footprint_size;
  point.y = footprint_size;
  footprint.push_back(point);
  point.x = -footprint_size;
  point.y = -footprint_size;
  footprint.push_back(point);
  point.x = footprint_size;
  point.y = -footprint_size;
  footprint.push_back(point);
  point.x = -footprint_size;
  point.y = footprint_size;
  footprint.push_back(point);

  costmap_ros->setRobotFootprint(footprint);

  // Add obstacle
  double obstacle_pose_x = 8.0;  // [m]
  double obstacle_pose_y = 8.0;  // [m]
  double obstacle_size = 0.4;    // [m]
  int obs_x = obstacle_pose_x / resolution;
  int obs_y = obstacle_pose_y / resolution;
  int obs_size = obstacle_size / resolution;
  unsigned char obs_cost = 250;
  addObstacle(costmap_ros->getCostmap(), obs_x, obs_y, obs_size, obs_cost);

  /**
   * Setup Optimizer
   **/
  std::vector<rclcpp::Parameter> params;
  rclcpp::NodeOptions options;
  std::string node_name = "MPPI";
  params.emplace_back(rclcpp::Parameter(node_name + ".iteration_count", iteration_count));
  params.emplace_back(rclcpp::Parameter(node_name + ".batch_size", batch_size));
  params.emplace_back(rclcpp::Parameter(node_name + ".time_steps", time_steps));
  params.emplace_back(rclcpp::Parameter(node_name + ".lookahead_dist", lookahead_dist));
  params.emplace_back(rclcpp::Parameter(node_name + ".motion_model", motion_model));
  params.emplace_back(rclcpp::Parameter(node_name + ".critics", critics));
  // params.emplace_back(rclcpp::Parameter(node_name +
  // ".ObstaclesCritic.enabled", false));
  params.emplace_back(rclcpp::Parameter("controller_frequency", controller_frequency));
  options.parameter_overrides(params);

  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(node_name, options);
  auto parameters_handler = std::make_unique<mppi::ParametersHandler>(node);
  auto optimizer = std::make_shared<mppi::Optimizer>();
  std::weak_ptr<rclcpp_lifecycle::LifecycleNode> weak_ptr_node{ node };
  optimizer->initialize(weak_ptr_node, node->get_name(), costmap_ros, parameters_handler.get());

  geometry_msgs::msg::PoseStamped start_pose;
  std::string frame = "odom";
  auto time = node->get_clock()->now();
  start_pose.header.frame_id = frame;
  start_pose.header.stamp = time;
  start_pose.pose.position.x = 5;
  start_pose.pose.position.y = 5;

  geometry_msgs::msg::Twist velocity;
  nav_msgs::msg::Path empty_path;
  auto goal_pose = start_pose;
  goal_pose.pose.position.x = 10;
  goal_pose.pose.position.y = 10;
  empty_path.poses.push_back(goal_pose);
  empty_path.header.frame_id = frame;
  empty_path.header.stamp = time;
  nav2_core::GoalChecker* dummy_goal_checker{ nullptr };
  std::cout << "About to optimize" << std::endl;
  int num_iterations = 10000;
  double total_duration_ms = 0.0f;
  for (int i = 0; i < num_iterations; i++)
  {
    auto start = std::chrono::steady_clock::now();
    optimizer->evalControl(start_pose, velocity, empty_path, dummy_goal_checker);
    auto end = std::chrono::steady_clock::now();
    total_duration_ms += (end - start).count() / 1e6;
  }
  std::cout << "Avg optimization time (ms): " << total_duration_ms / num_iterations << std::endl;
  rclcpp::shutdown();
  return 0;
}
