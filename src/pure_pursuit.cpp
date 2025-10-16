#include "pure_pursuit.hpp"

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

PurePursuit::PurePursuit() : Node("pure_pursuit_node") {
  // 초기 waypoint index (예시 값)
  waypoints.index = 125807609;

  // 파라미터 선언
  this->declare_parameter("waypoints_path",
                          "/sim_ws/src/pure_pursuit/racelines/"
                          "traj_race_cl-2025-02-07 06_48_41.623737.csv");
  this->declare_parameter("odom_topic", "/ego_racecar/odom");
  this->declare_parameter("car_refFrame", "ego_racecar/base_link");
  this->declare_parameter("drive_topic", "/drive");
  this->declare_parameter("rviz_current_waypoint_topic", "/current_waypoint");
  this->declare_parameter("rviz_lookahead_waypoint_topic",
                          "/lookahead_waypoint");
  this->declare_parameter("global_refFrame", "map");
  this->declare_parameter("min_lookahead", 0.5);
  this->declare_parameter("max_lookahead", 1.0);
  this->declare_parameter("lookahead_ratio", 8.0);
  this->declare_parameter("min_searching_idx_offset", 10);
  this->declare_parameter("max_searching_idx_offset", 40);
  this->declare_parameter("K_p", 0.5);
  this->declare_parameter("K_d", 0.1); // 미분 게인
  this->declare_parameter("K_i", 0.05); // 추가된 적분 게인
  this->declare_parameter("steering_limit", 25.0);
  this->declare_parameter("velocity_percentage", 0.6);

  // 파라미터 읽어오기
  waypoints_path = this->get_parameter("waypoints_path").as_string();
  odom_topic = this->get_parameter("odom_topic").as_string();
  car_refFrame = this->get_parameter("car_refFrame").as_string();
  drive_topic = this->get_parameter("drive_topic").as_string();
  rviz_current_waypoint_topic =
      this->get_parameter("rviz_current_waypoint_topic").as_string();
  rviz_lookahead_waypoint_topic =
      this->get_parameter("rviz_lookahead_waypoint_topic").as_string();
  global_refFrame = this->get_parameter("global_refFrame").as_string();
  min_lookahead = this->get_parameter("min_lookahead").as_double();
  max_lookahead = this->get_parameter("max_lookahead").as_double();
  lookahead_ratio = this->get_parameter("lookahead_ratio").as_double();
  min_searching_idx_offset =
      this->get_parameter("min_searching_idx_offset").as_int();
  max_searching_idx_offset =
      this->get_parameter("max_searching_idx_offset").as_int();
  K_p = this->get_parameter("K_p").as_double();
  K_d = this->get_parameter("K_d").as_double();
  K_i = this->get_parameter("K_i").as_double();  // I제어기 파라미터 읽기
  steering_limit = this->get_parameter("steering_limit").as_double();
  velocity_percentage = this->get_parameter("velocity_percentage").as_double();

  // 초기 적분 오차 초기화
  integral_error = 0.0;

  // Subscriber, Publisher, Timer 등 초기화
  subscription_odom = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, 25, std::bind(&PurePursuit::odom_callback, this, _1));
  timer_ = this->create_wall_timer(
      2000ms, std::bind(&PurePursuit::timer_callback, this));

  publisher_drive =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          drive_topic, 25);
  vis_current_point_pub =
      this->create_publisher<visualization_msgs::msg::Marker>(
          rviz_current_waypoint_topic, 10);
  vis_lookahead_point_pub =
      this->create_publisher<visualization_msgs::msg::Marker>(
          rviz_lookahead_waypoint_topic, 10);

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  transform_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  RCLCPP_INFO(this->get_logger(), "Pure pursuit node has been launched");

  load_waypoints();

  // PD 제어를 위한 초기값 설정
  prev_error = 0.0;
  prev_time = this->now();
}

double PurePursuit::to_radians(double degrees) {
  return degrees * M_PI / 180.0;
}

double PurePursuit::to_degrees(double radians) {
  return radians * 180.0 / M_PI;
}

double PurePursuit::p2pdist(double &x1, double &x2, double &y1, double &y2) {
  return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
}

void PurePursuit::load_waypoints() {
  csvFile_waypoints.open(waypoints_path, std::ios::in);

  if (!csvFile_waypoints.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot Open CSV File: %s",
                 waypoints_path.c_str());
    return;
  } else {
    RCLCPP_INFO(this->get_logger(), "CSV File Opened");
  }

  std::string line, word;
  while (!csvFile_waypoints.eof()) {
    std::getline(csvFile_waypoints, line, '\n');
    std::stringstream s(line);
    int j = 0;
    while (getline(s, word, ',')) {
      if (!word.empty()) {
        if (j == 0) {
          waypoints.X.push_back(std::stod(word));
        } else if (j == 1) {
          waypoints.Y.push_back(std::stod(word));
        } else if (j == 2) {
          waypoints.V.push_back(std::stod(word));
        }
      }
      j++;
    }
  }

  csvFile_waypoints.close();
  num_waypoints = waypoints.X.size();
  RCLCPP_INFO(this->get_logger(), "Finished loading %d waypoints from %s",
              num_waypoints, waypoints_path.c_str());

  double average_dist_between_waypoints = 0.0;
  for (int i = 0; i < num_waypoints - 1; i++) {
    average_dist_between_waypoints += p2pdist(
        waypoints.X[i], waypoints.X[i + 1], waypoints.Y[i], waypoints.Y[i + 1]);
  }
  average_dist_between_waypoints /= num_waypoints;
  RCLCPP_INFO(this->get_logger(), "Average distance between waypoints: %f",
              average_dist_between_waypoints);
}

void PurePursuit::visualize_lookahead_point(Eigen::Vector3d &point) {
  auto marker = visualization_msgs::msg::Marker();
  marker.header.frame_id = "map";
  marker.header.stamp = this->now();
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.25;
  marker.scale.y = 0.25;
  marker.scale.z = 0.25;
  marker.color.a = 1.0;
  marker.color.r = 1.0;

  marker.pose.position.x = point(0);
  marker.pose.position.y = point(1);
  marker.id = 1;
  vis_lookahead_point_pub->publish(marker);
}

void PurePursuit::visualize_current_point(Eigen::Vector3d &point) {
  auto marker = visualization_msgs::msg::Marker();
  marker.header.frame_id = "map";
  marker.header.stamp = this->now();
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.25;
  marker.scale.y = 0.25;
  marker.scale.z = 0.25;
  marker.color.a = 1.0;
  marker.color.b = 1.0;

  marker.pose.position.x = point(0);
  marker.pose.position.y = point(1);
  marker.id = 1;
  vis_current_point_pub->publish(marker);
}

int PurePursuit::path_idx_limiter(int idx) {
  for (idx; idx < 0; idx = +num_waypoints) {
  }
  return idx % num_waypoints;
}

void PurePursuit::get_waypoint_new() {
  if (waypoints.velocity_index < 0) { // first searching idx
    int closest_idx = 0;
    double closest_dist = p2pdist(waypoints.X[closest_idx], x_car_world,
                                  waypoints.Y[closest_idx], y_car_world);
    for (int i = 1; i < num_waypoints; i++) {
      double cur_dist =
          p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
      if (cur_dist < closest_dist) {
        closest_idx = i;
        closest_dist = cur_dist;
      }
    }
    waypoints.velocity_index = closest_idx;
  } else { // searching idx on warm starts(among the offset)
    int cur_idx = waypoints.velocity_index;
    int backIdx = path_idx_limiter(cur_idx - min_searching_idx_offset);
    int searching_counter = min_searching_idx_offset + max_searching_idx_offset;
    double min_dist = p2pdist(waypoints.X[cur_idx], x_car_world,
                              waypoints.Y[cur_idx], y_car_world);

    for (int i = 0; i < searching_counter; i++) {
      int searching_idx = path_idx_limiter(i + backIdx);
      double searching_dist = p2pdist(waypoints.X[searching_idx], x_car_world,
                                      waypoints.Y[searching_idx], y_car_world);
      if (searching_dist < min_dist) {
        min_dist = searching_dist;
        cur_idx = searching_idx;
      }
    }
    waypoints.velocity_index = cur_idx;
  }

  // lookahead calc
  double lookahead = std::min(
      std::max(min_lookahead, max_lookahead * curr_velocity / lookahead_ratio),
      max_lookahead);

  // lookahead_idx(waypoints.index) updater;
  int lookahead_searching_idx = waypoints.velocity_index;
  int next_lookahead_idx = path_idx_limiter(lookahead_searching_idx + 1);
  double cur_point_to_lookahead_dist = 0;
  while (cur_point_to_lookahead_dist < lookahead) {
    cur_point_to_lookahead_dist += p2pdist(
        waypoints.X[lookahead_searching_idx], waypoints.X[next_lookahead_idx],
        waypoints.Y[lookahead_searching_idx], waypoints.Y[next_lookahead_idx]);
    lookahead_searching_idx = next_lookahead_idx;
    next_lookahead_idx = path_idx_limiter(next_lookahead_idx + 1);
  }
  waypoints.index = lookahead_searching_idx;
}

void PurePursuit::get_waypoint() {
  double longest_distance = 0;
  int final_i = -1;
  int start = waypoints.index;
  RCLCPP_INFO(this->get_logger(), "start index: %d", start);
  int end = (waypoints.index + 50) % num_waypoints;

  double lookahead = std::min(
      std::max(min_lookahead, max_lookahead * curr_velocity / lookahead_ratio),
      max_lookahead);

  if (end < start) {
    for (int i = start; i < num_waypoints; i++) {
      double dist =
          p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
      if (dist <= lookahead && dist >= longest_distance) {
        longest_distance = dist;
        final_i = i;
      }
    }
    for (int i = 0; i < end; i++) {
      double dist =
          p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
      if (dist <= lookahead && dist >= longest_distance) {
        longest_distance = dist;
        final_i = i;
      }
    }
  } else {
    for (int i = start; i < end; i++) {
      double dist =
          p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
      if (dist <= lookahead && dist >= longest_distance) {
        longest_distance = dist;
        final_i = i;
      }
    }
  }

  if (final_i == -1) {
    final_i = 0;
    for (int i = 0; i < num_waypoints; i++) {
      double dist =
          p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
      if (dist <= lookahead && dist >= longest_distance) {
        longest_distance = dist;
        final_i = i;
      }
    }
  }

  double shortest_distance =
      p2pdist(waypoints.X[0], x_car_world, waypoints.Y[0], y_car_world);
  int velocity_i = 0;
  for (int i = 0; i < num_waypoints; i++) {
    double dist =
        p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
    if (dist <= shortest_distance) {
      shortest_distance = dist;
      velocity_i = i;
    }
  }

  waypoints.index = final_i;
  waypoints.velocity_index = velocity_i;
}

void PurePursuit::quat_to_rot(double q0, double q1, double q2, double q3) {
  double r00 = 2.0 * (q0 * q0 + q1 * q1) - 1.0;
  double r01 = 2.0 * (q1 * q2 - q0 * q3);
  double r02 = 2.0 * (q1 * q3 + q0 * q2);

  double r10 = 2.0 * (q1 * q2 + q0 * q3);
  double r11 = 2.0 * (q0 * q0 + q2 * q2) - 1.0;
  double r12 = 2.0 * (q2 * q3 - q0 * q1);

  double r20 = 2.0 * (q1 * q3 - q0 * q2);
  double r21 = 2.0 * (q2 * q3 + q0 * q1);
  double r22 = 2.0 * (q0 * q0 + q3 * q3) - 1.0;

  rotation_m << r00, r01, r02, r10, r11, r12, r20, r21, r22;
}

void PurePursuit::transformandinterp_waypoint() {
  // 현재 추종할 waypoint와 속도 프로파일용 waypoint 업데이트
  waypoints.lookahead_point_world << waypoints.X[waypoints.index],
      waypoints.Y[waypoints.index], 0.0;
  waypoints.current_point_world << waypoints.X[waypoints.velocity_index],
      waypoints.Y[waypoints.velocity_index], 0.0;

  visualize_lookahead_point(waypoints.lookahead_point_world);
  visualize_current_point(waypoints.current_point_world);

  geometry_msgs::msg::TransformStamped transformStamped;
  try {
    transformStamped = tf_buffer_->lookupTransform(
        car_refFrame, global_refFrame, tf2::TimePointZero);
  } catch (tf2::TransformException &ex) {
    RCLCPP_INFO(this->get_logger(), "Could not transform. Error: %s",
                ex.what());
  }

  Eigen::Vector3d translation_v(transformStamped.transform.translation.x,
                                transformStamped.transform.translation.y,
                                transformStamped.transform.translation.z);
  quat_to_rot(transformStamped.transform.rotation.w,
              transformStamped.transform.rotation.x,
              transformStamped.transform.rotation.y,
              transformStamped.transform.rotation.z);

  waypoints.lookahead_point_car =
      (rotation_m * waypoints.lookahead_point_world) + translation_v;
}

double PurePursuit::p_controller() {
  // lookahead point를 이용한 lateral error 계산
  double r = waypoints.lookahead_point_car.norm(); // sqrt(x^2 + y^2)
  double y = waypoints.lookahead_point_car(1);
  double error = 2.0 * y / (r * r);

  // 현재 시간과 이전 시간의 차이 (dt, 초 단위)
  rclcpp::Time current_time = this->now();
  double dt = (current_time - prev_time).seconds();

  // I제어기: 적분 오차 누적 (적분 windup에 대한 방지 처리는 필요에 따라 추가)
  integral_error += error * dt;

  double derivative = 0.0;
  if (dt > 0.0) {
    derivative = (error - prev_error) / dt;
  }

  // PID 제어: steering angle = K_p * error + K_i * integral_error + K_d * derivative
  double angle = K_p * error + K_i * integral_error + K_d * derivative;

  // 이전 오차 및 시간 업데이트
  prev_error = error;
  prev_time = current_time;

  return angle;
}

double PurePursuit::get_velocity(double steering_angle) {
  double velocity = 0;
  if (waypoints.V[waypoints.velocity_index]) {
    velocity = waypoints.V[waypoints.velocity_index] * velocity_percentage;
  } else {
    if (std::abs(steering_angle) >= to_radians(0.0) &&
        std::abs(steering_angle) < to_radians(10.0)) {
      velocity = 6.0 * velocity_percentage;
    } else if (std::abs(steering_angle) >= to_radians(10.0) &&
               std::abs(steering_angle) <= to_radians(20.0)) {
      velocity = 2.5 * velocity_percentage;
    } else {
      velocity = 2.0 * velocity_percentage;
    }
  }
  return velocity;
}

void PurePursuit::publish_message(double steering_angle) {
  auto drive_msgObj = ackermann_msgs::msg::AckermannDriveStamped();
  if (steering_angle < 0.0) {
    drive_msgObj.drive.steering_angle =
        std::max(steering_angle, -to_radians(steering_limit));
  } else {
    drive_msgObj.drive.steering_angle =
        std::min(steering_angle, to_radians(steering_limit));
  }

  curr_velocity = get_velocity(drive_msgObj.drive.steering_angle);
  drive_msgObj.drive.speed = curr_velocity;

  RCLCPP_INFO(this->get_logger(),
              "index: %d ... distance: %.2fm ... Speed: %.2fm/s ... Steering "
              "Angle: %.2f ... K_p: %.2f ... K_i: %.2f ... velocity_percentage: %.2f",
              waypoints.index,
              p2pdist(waypoints.X[waypoints.index], x_car_world,
                      waypoints.Y[waypoints.index], y_car_world),
              drive_msgObj.drive.speed,
              to_degrees(drive_msgObj.drive.steering_angle), K_p, K_i,
              velocity_percentage);

  publisher_drive->publish(drive_msgObj);
}

void PurePursuit::odom_callback(
    const nav_msgs::msg::Odometry::ConstSharedPtr odom_submsgObj) {
  x_car_world = odom_submsgObj->pose.pose.position.x;
  y_car_world = odom_submsgObj->pose.pose.position.y;
  RCLCPP_INFO(this->get_logger(), "odom x:  %.4f  %.4f", x_car_world,
              y_car_world);

  // waypoint 업데이트, tf 변환, 그리고 PID 제어를 통한 steering 계산
  get_waypoint_new();
  transformandinterp_waypoint();
  double steering_angle = p_controller();
  publish_message(steering_angle);
}

void PurePursuit::timer_callback() {
  // 주기적으로 파라미터 업데이트
  K_p = this->get_parameter("K_p").as_double();
  K_d = this->get_parameter("K_d").as_double();
  K_i = this->get_parameter("K_i").as_double(); // I제어기 파라미터 업데이트
  velocity_percentage = this->get_parameter("velocity_percentage").as_double();
  min_lookahead = this->get_parameter("min_lookahead").as_double();
  max_lookahead = this->get_parameter("max_lookahead").as_double();
  lookahead_ratio = this->get_parameter("lookahead_ratio").as_double();
  steering_limit = this->get_parameter("steering_limit").as_double();
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node_ptr = std::make_shared<PurePursuit>();
  rclcpp::spin(node_ptr);
  rclcpp::shutdown();
  return 0;
}
