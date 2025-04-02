#ifndef PURE_PURSUIT_HPP
#define PURE_PURSUIT_HPP

#include <math.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Eigen>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#define _USE_MATH_DEFINES
using std::placeholders::_1;
using namespace std::chrono_literals;

class PurePursuit : public rclcpp::Node {
public:
  PurePursuit();

private:
  // 구조체: CSV 파일에서 읽어온 waypoints 데이터
  struct csvFileData {
    std::vector<double> X;
    std::vector<double> Y;
    std::vector<double> V;

    int index;          // steering waypoint(lookahead)
    int velocity_index; // velocity get point(current point)

    Eigen::Vector3d lookahead_point_world; // 월드 좌표계 (보통 "map")
    Eigen::Vector3d lookahead_point_car;   // 차량 좌표계
    Eigen::Vector3d current_point_world; // 차량에 가장 가까운 waypoint (속도
                                         // 프로파일에 사용)
  };

  Eigen::Matrix3d rotation_m;

  double x_car_world;
  double y_car_world;

  std::string odom_topic;
  std::string car_refFrame;
  std::string drive_topic;
  std::string global_refFrame;
  std::string rviz_current_waypoint_topic;
  std::string rviz_lookahead_waypoint_topic;
  std::string waypoints_path;
  double K_p;
  double K_d; // PD 제어를 위한 미분(derivative) 게인
  double min_lookahead;
  double max_lookahead;
  double lookahead_ratio;
  double steering_limit;
  double velocity_percentage;
  double curr_velocity = 0.0;
  int min_searching_idx_offset;
  int max_searching_idx_offset;

  bool emergency_breaking = false;
  std::string lane_number = "left"; // "left" 또는 "right"

  // 파일 객체
  std::fstream csvFile_waypoints;

  // Waypoint 데이터 구조체
  csvFileData waypoints;
  int num_waypoints;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;

  // Subscriber
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_odom;

  // Publisher
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      publisher_drive;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      vis_current_point_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      vis_lookahead_point_pub;

  // TF 관련 포인터
  std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

  // PD 제어를 위한 이전 오차와 이전 시간 (미분 항 계산용)
  double prev_error;
  rclcpp::Time prev_time;

  // private 함수들
  double to_radians(double degrees);
  double to_degrees(double radians);
  double p2pdist(double &x1, double &x2, double &y1, double &y2);

  void load_waypoints();

  void visualize_lookahead_point(Eigen::Vector3d &point);
  void visualize_current_point(Eigen::Vector3d &point);

  void get_waypoint();

  void quat_to_rot(double q0, double q1, double q2, double q3);

  void transformandinterp_waypoint();

  double p_controller();

  double get_velocity(double steering_angle);

  void publish_message(double steering_angle);

  void
  odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_submsgObj);

  void timer_callback();

  void get_waypoint_new();

  int path_idx_limiter(int idx);
};

#endif // PURE_PURSUIT_HPP
