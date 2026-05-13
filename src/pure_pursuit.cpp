#include "pure_pursuit.hpp"

#include <cmath>

#include <Eigen/Eigen>
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"


namespace {
inline double planar_distance_sq(double x1, double y1, double x2, double y2) {
  const double dx = x2 - x1;
  const double dy = y2 - y1;
  return dx * dx + dy * dy;
}
} // namespace

PurePursuit::PurePursuit() : Node("pure_pursuit_node") {
  // 초기 lookahead index 값. 실제 사용은 waypoints.index 로 관리됩니다.
  lookahead_index = 0;

  // 파라미터 선언
  this->declare_parameter("odom_topic", "/ego_racecar/odom");
  this->declare_parameter("car_refFrame", "ego_racecar/base_link");
  this->declare_parameter("drive_topic", "/drive");
  this->declare_parameter("path_topic", "/Path");
  this->declare_parameter("rviz_current_waypoint_topic", "/current_waypoint");
  this->declare_parameter("rviz_lookahead_waypoint_topic",
                          "/lookahead_waypoint");
  this->declare_parameter("global_refFrame", "map");
  this->declare_parameter("path_is_circular", true);
  this->declare_parameter("min_lookahead", 0.5);
  this->declare_parameter("max_lookahead", 1.0);
  this->declare_parameter("lookahead_ratio", 8.0);
  this->declare_parameter("speed_lookahead_time", 0.0);
  this->declare_parameter("speed_lookahead_accel_limit", 8.0);
  this->declare_parameter("speed_lookahead_accel_weight", 1.0);
  this->declare_parameter("min_searching_idx_offset", 10);
  this->declare_parameter("max_searching_idx_offset", 40);
  this->declare_parameter("K_p", 0.5);
  this->declare_parameter("K_d", 0.1);  // 미분 게인
  this->declare_parameter("K_i", 0.05); // 추가된 적분 게인
  this->declare_parameter("steering_limit", 25.0);
  this->declare_parameter("velocity_percentage", 0.6);
  this->declare_parameter("heading_error_gain", 0.0);
  this->declare_parameter("steer_reduction_speed_threshold", 5.0);
  this->declare_parameter("steer_reduction_constant_coef", 0.85);
  this->declare_parameter("steer_reduction_linear_coef", 0.03);
  this->declare_parameter("steer_reduction_min_scale", 0.3);
  this->declare_parameter("speed_reduction_steer_angle_deg", 12.0);
  this->declare_parameter("max_allowed_steer_drop_deg", 5.0);
  this->declare_parameter("speed_reduction_adjust", 0.0);
  this->declare_parameter("speed_reduction_prev_scale", 0.0);
  this->declare_parameter("steering_expo_gain", 0.0);
  this->declare_parameter("steering_expo_curve", 2.0);
  this->declare_parameter("slow_with_obs", true);
  this->declare_parameter("obs_slow_th", 3.0);
  this->declare_parameter("obs_slow_percentage", 0.6);

  // 파라미터 읽어오기
  odom_topic = this->get_parameter("odom_topic").as_string();
  car_refFrame = this->get_parameter("car_refFrame").as_string();
  drive_topic = this->get_parameter("drive_topic").as_string();
  path_topic = this->get_parameter("path_topic").as_string();
  rviz_current_waypoint_topic =
      this->get_parameter("rviz_current_waypoint_topic").as_string();
  rviz_lookahead_waypoint_topic =
      this->get_parameter("rviz_lookahead_waypoint_topic").as_string();
  global_refFrame = this->get_parameter("global_refFrame").as_string();
  path_is_circular = this->get_parameter("path_is_circular").as_bool();
  min_lookahead = this->get_parameter("min_lookahead").as_double();
  max_lookahead = this->get_parameter("max_lookahead").as_double();
  lookahead_ratio = this->get_parameter("lookahead_ratio").as_double();
  speed_lookahead_time =
      this->get_parameter("speed_lookahead_time").as_double();
  speed_lookahead_accel_limit =
      this->get_parameter("speed_lookahead_accel_limit").as_double();
  speed_lookahead_accel_weight =
      this->get_parameter("speed_lookahead_accel_weight").as_double();
  min_searching_idx_offset =
      this->get_parameter("min_searching_idx_offset").as_int();
  max_searching_idx_offset =
      this->get_parameter("max_searching_idx_offset").as_int();
  K_p = this->get_parameter("K_p").as_double();
  K_d = this->get_parameter("K_d").as_double();
  K_i = this->get_parameter("K_i").as_double(); // I제어기 파라미터 읽기
  steering_limit = this->get_parameter("steering_limit").as_double();
  velocity_percentage = this->get_parameter("velocity_percentage").as_double();
  heading_error_gain =
      this->get_parameter("heading_error_gain").as_double();
  steer_reduction_speed_threshold =
      this->get_parameter("steer_reduction_speed_threshold").as_double();
  steer_reduction_constant_coef =
      this->get_parameter("steer_reduction_constant_coef").as_double();
  steer_reduction_linear_coef =
      this->get_parameter("steer_reduction_linear_coef").as_double();
  steer_reduction_min_scale =
      this->get_parameter("steer_reduction_min_scale").as_double();
  speed_reduction_angle_threshold = to_radians(
      this->get_parameter("speed_reduction_steer_angle_deg").as_double());
  max_allowed_steer_drop = to_radians(
      this->get_parameter("max_allowed_steer_drop_deg").as_double());
  speed_reduction_adjust =
      this->get_parameter("speed_reduction_adjust").as_double();
  speed_reduction_prev_scale =
      this->get_parameter("speed_reduction_prev_scale").as_double();
  steering_expo_gain =
      this->get_parameter("steering_expo_gain").as_double();
  steering_expo_curve =
      this->get_parameter("steering_expo_curve").as_double();
  slow_with_obs = 
      this->get_parameter("slow_with_obs").as_bool();
  slow_th_dist = 
      this->get_parameter("obs_slow_th").as_double();
  slow_amount = 
      this->get_parameter("obs_slow_percentage").as_double();

  // 초기 적분 오차 초기화
  integral_error = 0.0;
  x_car_world = 0.0;
  y_car_world = 0.0;
  car_heading = 0.0;
  previous_speed_reduction = 0.0;
  previous_odom_stamp = this->now();

  // 경로 수신 플래그 초기화
  path_received_ = false;

  // Subscriber, Publisher, Timer 등 초기화
  subscription_odom = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, 25, std::bind(&PurePursuit::odom_callback, this, _1));

  subscription_odom_obs = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "/static_obstacle", 10, std::bind(&PurePursuit::obs_odom_callback, this, _1));
  obs_status = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "/obj_flag", 10, std::bind(&PurePursuit::obs_status_callback, this, _1));

  timer_ = this->create_wall_timer(
      2000ms, std::bind(&PurePursuit::timer_callback, this));

  rclcpp::QoS pathQos(rclcpp::KeepLast(10));
  pathQos.reliability(rclcpp::ReliabilityPolicy::Reliable);
  //pathQos.durability(rclcpp::DurabilityPolicy::TransientLocal);

  subscription_path = this->create_subscription<nav_msgs::msg::Path>(
      path_topic, pathQos,
      std::bind(&PurePursuit::path_callback, this, std::placeholders::_1));

  publisher_drive =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          drive_topic, 25);
  vis_current_point_pub =
      this->create_publisher<visualization_msgs::msg::Marker>(
          rviz_current_waypoint_topic, 10);
  vis_lookahead_point_pub =
      this->create_publisher<visualization_msgs::msg::Marker>(
          rviz_lookahead_waypoint_topic, 10);

  RCLCPP_INFO(this->get_logger(), "Pure pursuit node has been launched");

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

double PurePursuit::p2pdist(const double &x1, const double &x2,
                            const double &y1, const double &y2) {
  return std::hypot(x2 - x1, y2 - y1);
}

// nav_msgs::Path 토픽을 통해 전달된 경로를 수신하고 내부 waypoints 구조체를
// 갱신합니다. Path 메시지의 각 PoseStamped에서 position.x, position.y,
// position.z를 각각 X, Y, V로 저장합니다.
void PurePursuit::path_callback(const nav_msgs::msg::Path::SharedPtr path_msg) {
  // 경로가 비어있으면 무시합니다.
  if (!path_msg || path_msg->poses.empty()) {
    RCLCPP_WARN(this->get_logger(),
                "Received empty path message, ignoring it.");
    return;
  }

  // 기존 waypoint 데이터 초기화
  waypoints.X.clear();
  waypoints.Y.clear();
  waypoints.V.clear();
  waypoints.index = 0;
  waypoints.velocity_index = -1;
  waypoints.speed_index = -1;

  // 새 경로의 포즈를 순회하면서 좌표와 속도(v)를 저장
  for (const auto &pose_stamped : path_msg->poses) {
    const auto &pos = pose_stamped.pose.position;
    waypoints.X.push_back(pos.x);
    waypoints.Y.push_back(pos.y);
    // Path 메시지에서 z 값은 속도 정보를 담고 있다고 가정
    waypoints.V.push_back(pos.z);
  }

  num_waypoints = static_cast<int>(waypoints.X.size());
  RCLCPP_INFO(this->get_logger(),
              "Received new path with %d waypoints from topic.",
              num_waypoints);

  // 경로를 정상적으로 수신했음을 표시
  path_received_ = num_waypoints > 0;
}

void PurePursuit::visualize_lookahead_point(Eigen::Vector3d &point) {
  auto marker = visualization_msgs::msg::Marker();
  marker.header.frame_id = global_refFrame;
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
  marker.header.frame_id = global_refFrame;
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
  if (num_waypoints <= 0) {
    return 0;
  }

  if (path_is_circular) {
    const int mod = idx % num_waypoints;
    return mod >= 0 ? mod : mod + num_waypoints;
  }

  if (idx < 0) {
    return 0;
  }

  if (idx >= num_waypoints) {
    return num_waypoints - 1;
  }

  return idx;
}

int PurePursuit::advance_index_by_distance(int start_idx, double distance) {
  if (num_waypoints <= 1 || std::abs(distance) <= 1e-6) {
    return path_idx_limiter(start_idx);
  }

  int idx = path_idx_limiter(start_idx);
  const int direction = distance >= 0.0 ? 1 : -1;
  double remaining_distance = std::abs(distance);

  if (path_is_circular) {
    double path_length = 0.0;
    for (int i = 0; i < num_waypoints; ++i) {
      const int next_idx = path_idx_limiter(i + 1);
      path_length += p2pdist(waypoints.X[i], waypoints.X[next_idx],
                             waypoints.Y[i], waypoints.Y[next_idx]);
    }
    if (path_length > 1e-6) {
      remaining_distance = std::fmod(remaining_distance, path_length);
    }
  }

  const int max_segments = path_is_circular ? num_waypoints : num_waypoints - 1;
  int segments_checked = 0;
  while (remaining_distance > 1e-6 && segments_checked < max_segments) {
    const int next_idx =
        path_is_circular ? path_idx_limiter(idx + direction) : idx + direction;
    if (!path_is_circular && (next_idx < 0 || next_idx >= num_waypoints)) {
      break;
    }

    const double segment_distance =
        p2pdist(waypoints.X[idx], waypoints.X[next_idx], waypoints.Y[idx],
                waypoints.Y[next_idx]);
    idx = next_idx;
    ++segments_checked;

    if (segment_distance <= 1e-6) {
      continue;
    }
    remaining_distance -= segment_distance;
  }

  return path_idx_limiter(idx);
}

double PurePursuit::get_speed_lookahead_distance() const {
  const double t = speed_lookahead_time;
  const double accel_weight = std::max(0.0, speed_lookahead_accel_weight);
  double distance =
      current_longitudinal_speed * t +
      0.5 * accel_weight * longitudinal_acceleration * t * t;

  const double weighted_acceleration =
      accel_weight * longitudinal_acceleration;
  if (t > 0.0 && weighted_acceleration < 0.0) {
    const double predicted_speed =
        current_longitudinal_speed + weighted_acceleration * t;
    if (predicted_speed < 0.0) {
      const double stop_time =
          current_longitudinal_speed / -weighted_acceleration;
      distance = current_longitudinal_speed * stop_time +
                 0.5 * weighted_acceleration * stop_time * stop_time;
    }
  }

  if (t >= 0.0) {
    return std::max(0.0, distance);
  }
  return std::min(0.0, distance);
}

void PurePursuit::get_waypoint_new() {
  if (waypoints.velocity_index < 0) { // 첫 번째 호출 시: 가장 가까운 waypoint 찾기
    int closest_idx = 0;
    double closest_dist_sq =
        planar_distance_sq(waypoints.X[closest_idx], waypoints.Y[closest_idx],
                           x_car_world, y_car_world);
    for (int i = 1; i < num_waypoints; i++) {
      double cur_dist_sq =
          planar_distance_sq(waypoints.X[i], waypoints.Y[i], x_car_world,
                             y_car_world);
      if (cur_dist_sq < closest_dist_sq) {
        closest_idx = i;
        closest_dist_sq = cur_dist_sq;
      }
    }
    waypoints.velocity_index = closest_idx;
  } else {
    // warm start: 현재 velocity_index 부근에서 일정 범위 내에서 가장 가까운 포인트 탐색
    int cur_idx = waypoints.velocity_index;
    const int searching_counter =
        min_searching_idx_offset + max_searching_idx_offset;
    double min_dist_sq = planar_distance_sq(
        waypoints.X[cur_idx], waypoints.Y[cur_idx], x_car_world, y_car_world);

    if (path_is_circular) {
      int backIdx = path_idx_limiter(cur_idx - min_searching_idx_offset);
      for (int i = 0; i < searching_counter; i++) {
        int searching_idx = path_idx_limiter(backIdx + i);
        double searching_dist_sq =
            planar_distance_sq(waypoints.X[searching_idx],
                               waypoints.Y[searching_idx], x_car_world,
                               y_car_world);
        if (searching_dist_sq < min_dist_sq) {
          min_dist_sq = searching_dist_sq;
          cur_idx = searching_idx;
        }
      }
    } else {
      const int start_idx = std::max(cur_idx - min_searching_idx_offset, 0);
      const int end_idx =
          std::min(cur_idx + max_searching_idx_offset, num_waypoints - 1);
      for (int searching_idx = start_idx; searching_idx <= end_idx;
           ++searching_idx) {
        double searching_dist_sq =
            planar_distance_sq(waypoints.X[searching_idx],
                               waypoints.Y[searching_idx], x_car_world,
                               y_car_world);
        if (searching_dist_sq < min_dist_sq) {
          min_dist_sq = searching_dist_sq;
          cur_idx = searching_idx;
        }
      }
    }
    waypoints.velocity_index = cur_idx;
  }

  // lookahead calc
  double lookahead = std::min(
      std::max(min_lookahead, max_lookahead * curr_velocity / lookahead_ratio),
      max_lookahead);

  // lookahead_idx(waypoints.index) updater
  int lookahead_searching_idx = waypoints.velocity_index;
  double cur_point_to_lookahead_dist = 0.0;
  int steps_taken = 0;
  while (cur_point_to_lookahead_dist < lookahead) {
    int next_lookahead_idx;
    if (path_is_circular) {
      if (steps_taken >= num_waypoints) {
        // 이미 전체 경로만큼 탐색했으므로 더 이상 진행하지 않습니다.
        break;
      }
      ++steps_taken;
      next_lookahead_idx = path_idx_limiter(lookahead_searching_idx + 1);
      if (next_lookahead_idx == lookahead_searching_idx) {
        // 유효한 다음 포인트가 없는 경우
        break;
      }
    } else {
      if (lookahead_searching_idx >= num_waypoints - 1) {
        // 선형 경로의 끝에 도달
        break;
      }
      next_lookahead_idx = lookahead_searching_idx + 1;
    }

    const double segment_distance =
        p2pdist(waypoints.X[lookahead_searching_idx],
                waypoints.X[next_lookahead_idx],
                waypoints.Y[lookahead_searching_idx],
                waypoints.Y[next_lookahead_idx]);
    if (segment_distance <= 1e-6) {
      // 매우 짧거나 동일한 포인트인 경우: circular path에서는 다음 포인트로 넘어가며,
      // linear path에서는 더 이상 진행하지 않습니다.
      if (!path_is_circular) {
        break;
      }
      lookahead_searching_idx = next_lookahead_idx;
      continue;
    }

    cur_point_to_lookahead_dist += segment_distance;
    lookahead_searching_idx = next_lookahead_idx;

    if (!path_is_circular && lookahead_searching_idx >= num_waypoints - 1) {
      break;
    }
  }
  waypoints.index = lookahead_searching_idx;

  const double speed_lookahead_distance = get_speed_lookahead_distance();
  waypoints.speed_index =
      advance_index_by_distance(waypoints.velocity_index,
                                speed_lookahead_distance);
}

void PurePursuit::get_waypoint() {
  int final_i = -1;
  int start = waypoints.index;
  int end = (waypoints.index + 50) % num_waypoints;

  double lookahead = std::min(
      std::max(min_lookahead, max_lookahead * curr_velocity / lookahead_ratio),
      max_lookahead);
  const double lookahead_sq = lookahead * lookahead;
  double longest_distance_sq = 0.0;

  if (end < start) {
    for (int i = start; i < num_waypoints; i++) {
      double dist_sq =
          planar_distance_sq(waypoints.X[i], waypoints.Y[i], x_car_world,
                             y_car_world);
      if (dist_sq <= lookahead_sq && dist_sq >= longest_distance_sq) {
        longest_distance_sq = dist_sq;
        final_i = i;
      }
    }
    for (int i = 0; i < end; i++) {
      double dist_sq =
          planar_distance_sq(waypoints.X[i], waypoints.Y[i], x_car_world,
                             y_car_world);
      if (dist_sq <= lookahead_sq && dist_sq >= longest_distance_sq) {
        longest_distance_sq = dist_sq;
        final_i = i;
      }
    }
  } else {
    for (int i = start; i < end; i++) {
      double dist_sq =
          planar_distance_sq(waypoints.X[i], waypoints.Y[i], x_car_world,
                             y_car_world);
      if (dist_sq <= lookahead_sq && dist_sq >= longest_distance_sq) {
        longest_distance_sq = dist_sq;
        final_i = i;
      }
    }
  }

  if (final_i == -1) {
    final_i = 0;
    for (int i = 0; i < num_waypoints; i++) {
      double dist_sq =
          planar_distance_sq(waypoints.X[i], waypoints.Y[i], x_car_world,
                             y_car_world);
      if (dist_sq <= lookahead_sq && dist_sq >= longest_distance_sq) {
        longest_distance_sq = dist_sq;
        final_i = i;
      }
    }
  }

  double shortest_distance_sq =
      planar_distance_sq(waypoints.X[0], waypoints.Y[0], x_car_world,
                         y_car_world);
  int velocity_i = 0;
  for (int i = 0; i < num_waypoints; i++) {
    double dist_sq =
        planar_distance_sq(waypoints.X[i], waypoints.Y[i], x_car_world,
                           y_car_world);
    if (dist_sq <= shortest_distance_sq) {
      shortest_distance_sq = dist_sq;
      velocity_i = i;
    }
  }

  waypoints.index = final_i;
  waypoints.velocity_index = velocity_i;
}

double PurePursuit::apply_steering_expo(double steering_angle,
                                        double steering_limit_rad) {
  if (steering_limit_rad <= 1e-6) {
    return steering_angle;
  }

  const double limited_angle =
      std::clamp(steering_angle, -steering_limit_rad, steering_limit_rad);
  const double normalized = limited_angle / steering_limit_rad;
  const double abs_normalized = std::abs(normalized);
  const double curve = std::max(steering_expo_curve, 1e-6);
  const double expo_weight = std::pow(abs_normalized, curve);
  const double expo_scale =
      std::max(0.0, 1.0 + steering_expo_gain * expo_weight);

  return std::clamp(normalized * expo_scale, -1.0, 1.0) *
         steering_limit_rad;
}

void PurePursuit::transformandinterp_waypoint() {
  // 현재 추종할 waypoint와 속도 프로파일용 waypoint 업데이트
  // waypoints.index 는 lookahead 인덱스, waypoints.velocity_index 는 차량에 가장
  // 가까운 waypoint 인덱스입니다.
  if (num_waypoints == 0) {
    return;
  }
  // 경계 체크
  int look_idx = path_idx_limiter(waypoints.index);
  int vel_idx = path_idx_limiter(waypoints.velocity_index);
  // World 좌표계의 타겟 포인트 지정
  lookahead_point_world << waypoints.X[look_idx], waypoints.Y[look_idx], 0.0;
  current_point_world << waypoints.X[vel_idx], waypoints.Y[vel_idx], 0.0;

  // RViz 시각화를 위해 현재 포인트와 lookahead 포인트를 퍼블리시
  visualize_lookahead_point(lookahead_point_world);
  visualize_current_point(current_point_world);

  const double dx = lookahead_point_world(0) - x_car_world;
  const double dy = lookahead_point_world(1) - y_car_world;
  const double cos_yaw = std::cos(car_heading);
  const double sin_yaw = std::sin(car_heading);

  lookahead_point_car << cos_yaw * dx + sin_yaw * dy,
      -sin_yaw * dx + cos_yaw * dy, 0.0;
}

double PurePursuit::p_controller() {
  // lookahead point를 이용한 lateral error 계산
  // 차량 좌표계에서의 lookahead 포인트를 이용하여 lateral error 계산
  constexpr double kMinRadiusSq = 1e-6;
  const double y = lookahead_point_car(1);
  const double r_squared = lookahead_point_car.head<2>().squaredNorm();
  double lateral_error = 0.0;
  if (r_squared > kMinRadiusSq) {
    lateral_error = 2.0 * y / r_squared;
  }

  // 차량 주행 방향과 lookahead 방향의 각도 차이
  double heading_error = 0.0;
  if (num_waypoints > 0) {
    const double target_heading = std::atan2(
        lookahead_point_world(1) - y_car_world,
        lookahead_point_world(0) - x_car_world);
    heading_error = normalize_angle(target_heading - car_heading);
  }

  const double combined_error =
      lateral_error + heading_error_gain * heading_error;

  // 현재 시간과 이전 시간의 차이 (dt, 초 단위)
  rclcpp::Time current_time = this->now();
  double dt = (current_time - prev_time).seconds();

  // I제어기: 적분 오차 누적 (적분 windup에 대한 방지 처리는 필요에 따라 추가)
  integral_error += combined_error * dt;

  double derivative = 0.0;
  if (dt > 0.0) {
    derivative = (combined_error - prev_error) / dt;
  }

  // PID 제어: steering angle = K_p * error + K_i * integral_error + K_d *
  // derivative
  double angle =
      K_p * combined_error + K_i * integral_error + K_d * derivative;

  // 이전 오차 및 시간 업데이트
  prev_error = combined_error;
  prev_time = current_time;

  return angle;
}

double PurePursuit::get_velocity(double steering_angle) {
  double velocity = 0.0;
  const int speed_idx =
      waypoints.speed_index >= 0 ? path_idx_limiter(waypoints.speed_index)
                                 : path_idx_limiter(waypoints.velocity_index);
  // waypoint 메시지에서 속도 정보가 제공될 수 있으므로 우선 사용한다.
  if (!waypoints.V.empty() && speed_idx >= 0 &&
      speed_idx < static_cast<int>(waypoints.V.size()) &&
      waypoints.V[speed_idx] > 0.0) {
    velocity = waypoints.V[speed_idx] * velocity_percentage * velocity_reduce_obs;
  } else {
    // 속도 정보가 없으면 steering 각도에 따른 기본 속도를 설정한다.
    double abs_angle = std::abs(steering_angle);
    if (abs_angle < to_radians(10.0)) {
      velocity = 6.0 * velocity_percentage;
    } else if (abs_angle <= to_radians(20.0)) {
      velocity = 2.5 * velocity_percentage;
    } else {
      velocity = 2.0 * velocity_percentage;
    }
  }
  return velocity;
}

void PurePursuit::publish_message(double steering_angle) {
  auto drive_msgObj = ackermann_msgs::msg::AckermannDriveStamped();
  const double raw_steering = steering_angle;
  const double steering_limit_rad = to_radians(steering_limit);
  const double steering_for_velocity =
      std::clamp(raw_steering, -steering_limit_rad, steering_limit_rad);
  const double base_desired_speed = get_velocity(steering_for_velocity);
  double desired_speed = base_desired_speed;
  const double min_scale = std::clamp(steer_reduction_min_scale, 0.0, 1.0);
  const double positive_linear =
      std::max(steer_reduction_linear_coef, 0.0);
  const double prev_adjust =
      std::max(0.0, previous_speed_reduction) *
      std::max(0.0, speed_reduction_prev_scale);
  const double total_speed_adjust = speed_reduction_adjust + prev_adjust;
  const double adjusted_drop_speed =
      std::max(0.0, steer_reduction_speed_threshold - total_speed_adjust);

  auto compute_scale = [&](double speed) {
    double scale = 1.0;
    if (speed > steer_reduction_speed_threshold) {
      const double over_speed = speed - steer_reduction_speed_threshold;
      const double candidate =
          steer_reduction_constant_coef - positive_linear * over_speed;
      scale = std::clamp(candidate, min_scale, 1.0);
    }
    return scale;
  };

  double steer_scale = compute_scale(desired_speed);
  double adjusted_steering = raw_steering * steer_scale;
  const double original_abs = std::abs(raw_steering);
  auto calc_drop = [&](double scaled) {
    return std::max(0.0, original_abs - std::abs(scaled));
  };
  double steer_drop = calc_drop(adjusted_steering);

  if (original_abs >= speed_reduction_angle_threshold &&
      steer_drop > max_allowed_steer_drop) {
    const double safe_original =
        std::max(original_abs, 1e-6);
    double required_scale =
        1.0 - max_allowed_steer_drop / safe_original;
    required_scale = std::clamp(required_scale, min_scale, 1.0);

    const double scale_at_threshold =
        std::clamp(steer_reduction_constant_coef, min_scale, 1.0);
    double speed_cap = steer_reduction_speed_threshold;
    if (positive_linear > 1e-6 && required_scale < scale_at_threshold) {
      const double allowed_over_speed =
          (scale_at_threshold - required_scale) / positive_linear;
      speed_cap = steer_reduction_speed_threshold + allowed_over_speed;
    }
    double tuned_speed_cap =
        std::max(0.0, speed_cap - total_speed_adjust);
    desired_speed = std::min(desired_speed, tuned_speed_cap);
    steer_scale = compute_scale(desired_speed);
    adjusted_steering = raw_steering * steer_scale;
    steer_drop = calc_drop(adjusted_steering);

    if (steer_drop > max_allowed_steer_drop &&
        desired_speed > adjusted_drop_speed) {
      desired_speed = adjusted_drop_speed;
      steer_scale = compute_scale(desired_speed);
      adjusted_steering = raw_steering * steer_scale;
      steer_drop = calc_drop(adjusted_steering);
    }

    if (steer_drop > max_allowed_steer_drop) {
      desired_speed =
          std::max(0.0, steer_reduction_speed_threshold - total_speed_adjust);
      steer_scale = compute_scale(desired_speed);
      adjusted_steering = raw_steering * steer_scale;
      steer_drop = calc_drop(adjusted_steering);
    }
  }

  const double expo_steering =
      apply_steering_expo(adjusted_steering, steering_limit_rad);
  drive_msgObj.drive.steering_angle =
      std::clamp(expo_steering, -steering_limit_rad, steering_limit_rad);
  const double expo_scale =
      std::abs(adjusted_steering) > 1e-6
          ? std::abs(drive_msgObj.drive.steering_angle) /
                std::abs(adjusted_steering)
          : 1.0;

  curr_velocity = desired_speed;
  drive_msgObj.drive.speed = curr_velocity;
  previous_speed_reduction =
      std::max(0.0, base_desired_speed - curr_velocity);

  RCLCPP_INFO(
      this->get_logger(),
      "index: %d ... distance: %.2fm ... Speed: %.2fm/s ... Steering "
      "Angle: %.2f ... Raw Steering: %.2f ... SpeedIdx: %d ... "
      "SpeedLA: %.2fs/%.2fm ... Vx: %.2fm/s ... Ax: %.2fm/s^2 ... "
      "AxWeight: %.2f ... "
      "Steer Scale: %.2f ... "
      "Expo Scale: %.2f ... "
      "SpeedAdj: %.2f ... PrevReduction: %.2f ... K_p: %.2f "
      "... K_i: %.2f ... velocity_percentage: %.2f",
      waypoints.index,
      p2pdist(waypoints.X[waypoints.index], x_car_world,
              waypoints.Y[waypoints.index], y_car_world),
      drive_msgObj.drive.speed, to_degrees(drive_msgObj.drive.steering_angle),
      to_degrees(raw_steering), waypoints.speed_index,
      speed_lookahead_time, get_speed_lookahead_distance(),
      current_longitudinal_speed, longitudinal_acceleration,
      speed_lookahead_accel_weight, steer_scale, expo_scale, total_speed_adjust,
      previous_speed_reduction, K_p, K_i, velocity_percentage);

  publisher_drive->publish(drive_msgObj);
}

void PurePursuit::update_longitudinal_motion_state(
    const nav_msgs::msg::Odometry::ConstSharedPtr odom_submsgObj) {
  const auto &stamp_msg = odom_submsgObj->header.stamp;
  const bool has_stamp = stamp_msg.sec != 0 || stamp_msg.nanosec != 0;
  const rclcpp::Time stamp = has_stamp ? rclcpp::Time(stamp_msg) : this->now();

  double dt = 0.0;
  if (has_previous_odom_pose) {
    dt = (stamp - previous_odom_stamp).seconds();
  }

  double pose_based_speed = 0.0;
  if (has_previous_odom_pose && dt > 1e-4) {
    const double dx = x_car_world - previous_odom_x;
    const double dy = y_car_world - previous_odom_y;
    pose_based_speed =
        (std::cos(car_heading) * dx + std::sin(car_heading) * dy) / dt;
  }

  const double twist_speed = odom_submsgObj->twist.twist.linear.x;
  double measured_speed =
      std::abs(twist_speed) > 1e-3 ? twist_speed : pose_based_speed;
  measured_speed = std::max(0.0, measured_speed);

  if (has_previous_longitudinal_speed && dt > 1e-4) {
    double raw_acceleration =
        (measured_speed - previous_longitudinal_speed) / dt;
    const double accel_limit = std::abs(speed_lookahead_accel_limit);
    if (accel_limit > 1e-6) {
      raw_acceleration =
          std::clamp(raw_acceleration, -accel_limit, accel_limit);
    }
    longitudinal_acceleration = raw_acceleration;
  }

  current_longitudinal_speed = measured_speed;
  previous_longitudinal_speed = measured_speed;
  has_previous_longitudinal_speed = true;
  previous_odom_x = x_car_world;
  previous_odom_y = y_car_world;
  previous_odom_stamp = stamp;
  has_previous_odom_pose = true;
}

void PurePursuit::odom_callback(
    const nav_msgs::msg::Odometry::ConstSharedPtr odom_submsgObj) {
  x_car_world = odom_submsgObj->pose.pose.position.x;
  y_car_world = odom_submsgObj->pose.pose.position.y;
  const auto &orientation = odom_submsgObj->pose.pose.orientation;
  tf2::Quaternion q(orientation.x, orientation.y, orientation.z,
                    orientation.w);
  car_heading = tf2::getYaw(q);
  update_longitudinal_motion_state(odom_submsgObj);

  if (!global_refFrame.empty() &&
      odom_submsgObj->header.frame_id != global_refFrame) {
    RCLCPP_WARN_ONCE(
        this->get_logger(),
        "Expected odom frame '%s' but received '%s'. Using received frame.",
        global_refFrame.c_str(), odom_submsgObj->header.frame_id.c_str());
  }
  if (!car_refFrame.empty() && !odom_submsgObj->child_frame_id.empty() &&
      odom_submsgObj->child_frame_id != car_refFrame) {
    RCLCPP_WARN_ONCE(
        this->get_logger(),
        "Expected odom child frame '%s' but received '%s'. Using received "
        "frame.",
        car_refFrame.c_str(), odom_submsgObj->child_frame_id.c_str());
  }
  RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                        "odom x: %.4f y: %.4f", x_car_world, y_car_world);

  // 경로가 아직 수신되지 않았다면 제어를 수행하지 않고 대기합니다.
  if (!path_received_ || num_waypoints == 0) {
    RCLCPP_INFO(this->get_logger(),
                "No path received yet, waiting for path... (current odom will be ignored)");
    return;
  }

  // waypoint 업데이트, odom 기준 좌표 변환, 그리고 PID 제어를 통한 steering 계산
  get_waypoint_new();
  transformandinterp_waypoint();
  double steering_angle = p_controller();
  publish_message(steering_angle);
}

void PurePursuit::obs_odom_callback(const geometry_msgs::msg::PointStamped msg){
  x_obs = msg.point.x;
  y_obs = msg.point.y;
}

void PurePursuit::obs_status_callback(const geometry_msgs::msg::PointStamped msg){ //장애물 정보 반영 속도 줄이기
  int obsIsValid = int(msg.point.x);
  if(slow_with_obs && obsIsValid){
    if(p2pdist(x_obs, x_car_world, y_obs, y_car_world)<slow_th_dist){
      velocity_reduce_obs = slow_amount;
    }
    else{
      velocity_reduce_obs = 1;
    }
  }
  else{
    velocity_reduce_obs = 1;
  }

}

void PurePursuit::timer_callback() {
  // 주기적으로 파라미터 업데이트
  K_p = this->get_parameter("K_p").as_double();
  K_d = this->get_parameter("K_d").as_double();
  K_i = this->get_parameter("K_i").as_double(); // I제어기 파라미터 업데이트
  heading_error_gain =
      this->get_parameter("heading_error_gain").as_double();
  velocity_percentage = this->get_parameter("velocity_percentage").as_double();
  min_lookahead = this->get_parameter("min_lookahead").as_double();
  max_lookahead = this->get_parameter("max_lookahead").as_double();
  lookahead_ratio = this->get_parameter("lookahead_ratio").as_double();
  speed_lookahead_time =
      this->get_parameter("speed_lookahead_time").as_double();
  speed_lookahead_accel_limit =
      this->get_parameter("speed_lookahead_accel_limit").as_double();
  speed_lookahead_accel_weight =
      this->get_parameter("speed_lookahead_accel_weight").as_double();
  steering_limit = this->get_parameter("steering_limit").as_double();
  steering_expo_gain =
      this->get_parameter("steering_expo_gain").as_double();
  steering_expo_curve =
      this->get_parameter("steering_expo_curve").as_double();
  steer_reduction_speed_threshold =
      this->get_parameter("steer_reduction_speed_threshold").as_double();
  steer_reduction_constant_coef =
      this->get_parameter("steer_reduction_constant_coef").as_double();
  steer_reduction_linear_coef =
      this->get_parameter("steer_reduction_linear_coef").as_double();
  steer_reduction_min_scale =
      this->get_parameter("steer_reduction_min_scale").as_double();
  speed_reduction_angle_threshold = to_radians(
      this->get_parameter("speed_reduction_steer_angle_deg").as_double());
  max_allowed_steer_drop = to_radians(
      this->get_parameter("max_allowed_steer_drop_deg").as_double());
  speed_reduction_adjust =
      this->get_parameter("speed_reduction_adjust").as_double();
  speed_reduction_prev_scale =
      this->get_parameter("speed_reduction_prev_scale").as_double();
}

double PurePursuit::normalize_angle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node_ptr = std::make_shared<PurePursuit>();
  rclcpp::spin(node_ptr);
  rclcpp::shutdown();
  return 0;
}
