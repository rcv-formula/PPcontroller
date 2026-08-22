#include "pure_pursuit.hpp"

#include <cmath>

#include <Eigen/Eigen>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "std_msgs/msg/u_int16_multi_array.hpp"
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

// 런타임(ros2 param set)에 변경 가능한 파라미터 목록.
// apply_runtime_parameter()의 분기와 1:1로 대응합니다.
constexpr const char *kRuntimeParameterNames[] = {
    "drive_topic",
    "test_mode",
    "drive_test_topic",
    "rviz_runtime_params_x",
    "rviz_runtime_params_y",
    "rviz_runtime_params_z",
    "rf_speed_scale_channel",
    "rf_max_limit_channel",
    "rf_enable_channel",
    "rf_enable_threshold",
    "rf_value_min",
    "rf_value_max",
    "K_p",
    "K_d",
    "K_i",
    "heading_error_gain",
    "velocity_percentage",
    "max_speed_limit_percentage",
    "min_lookahead",
    "max_lookahead",
    "lookahead_ratio",
    "speed_profile_distance_offset",
    "steering_limit",
    "steering_expo_gain",
    "steering_expo_curve",
    "steer_reduction_speed_threshold",
    "steer_reduction_constant_coef",
    "steer_reduction_linear_coef",
    "steer_reduction_min_scale",
    "speed_reduction_steer_angle_deg",
    "max_allowed_steer_drop_deg",
    "speed_reduction_adjust",
    "speed_reduction_prev_scale",
    "drive_output_rate_hz",
    "publish_drive_on_odom",
    "visualization_rate_hz",
    "steer_latest_blend",
    "steer_large_change_blend",
    "steer_blend_change_threshold_deg",
    "steer_speed_filter_start_speed",
    "steer_speed_filter_end_speed",
    "steer_speed_filter_final_blend",
    "speed_latest_blend",
    "min_searching_idx_offset",
    "max_searching_idx_offset",
    "slow_with_obs",
    "obs_slow_th",
    "obs_slow_percentage",
    "speed_to_erpm_gain",
    "speed_to_erpm_offset",
    "wheel_speed_deadband",
    "wheel_speed_scale",
    "wheel_speed_timeout",
    "launch_start_enabled",
    "launch_start_channel",
    "launch_start_channel_threshold",
    "launch_start_engage_diff",
    "launch_start_release_diff",
    "launch_start_accel",
};

// 런타임 변경 시 허용 범위.
// 여기에 없는 숫자 파라미터는 유한값(NaN/inf 불가) 여부만 검사합니다.
// 사용 지점에서 이미 clamp하는 값(각종 blend, min_scale 등)은 등록하지 않습니다.
constexpr double kUnbounded = std::numeric_limits<double>::infinity();

struct RuntimeParameterBound {
  const char *name;
  double min_value;
  double max_value;
};

constexpr RuntimeParameterBound kRuntimeParameterBounds[] = {
    {"velocity_percentage", 0.0, 1.0},
    {"max_speed_limit_percentage", 0.0, 1.0},
    // 음수 게인은 제어 방향이 뒤집혀 즉시 발산합니다.
    {"K_p", 0.0, kUnbounded},
    {"K_i", 0.0, kUnbounded},
    {"K_d", 0.0, kUnbounded},
    // 0 이하이면 std::clamp(x, -lim, lim)의 lo > hi가 되어 정의되지 않은 동작
    {"steering_limit", 1e-3, 90.0},
    {"min_lookahead", 0.0, kUnbounded},
    {"max_lookahead", 0.0, kUnbounded},
    // lookahead 계산에서 나눗셈 분모로 사용됩니다.
    {"lookahead_ratio", 1e-6, kUnbounded},
    {"steering_expo_curve", 1e-6, kUnbounded},
    {"drive_output_rate_hz", 0.0, kUnbounded},
    {"visualization_rate_hz", 0.0, kUnbounded},
    {"obs_slow_th", 0.0, kUnbounded},
    {"obs_slow_percentage", 0.0, 1.0},
    {"rf_speed_scale_channel", 0.0, kUnbounded},
    {"rf_max_limit_channel", 0.0, kUnbounded},
    {"rf_enable_channel", 0.0, kUnbounded},
    {"launch_start_channel", 0.0, kUnbounded},
    {"launch_start_channel_threshold", 0.0, kUnbounded},
    {"wheel_speed_deadband", 0.0, kUnbounded},
    // 0이면 휠 속도가 항상 0으로 보여 래치가 계속 걸립니다.
    {"wheel_speed_scale", 1e-6, kUnbounded},
    {"wheel_speed_timeout", 0.0, kUnbounded},
    {"launch_start_engage_diff", 0.0, kUnbounded},
    {"launch_start_release_diff", 0.0, kUnbounded},
    // 0이면 램프가 움직이지 않아 래치가 풀리지 않습니다.
    {"launch_start_accel", 1e-3, kUnbounded},
};
} // namespace

PurePursuit::PurePursuit() : Node("pure_pursuit_node") {
  // 파라미터 선언
  this->declare_parameter("odom_topic", "/ego_racecar/odom");
  this->declare_parameter("car_refFrame", "ego_racecar/base_link");
  this->declare_parameter("drive_topic", "/drive");
  this->declare_parameter("test_mode", false);
  this->declare_parameter("drive_test_topic", "/drive_test");
  this->declare_parameter("path_topic", "/Path");
  this->declare_parameter("rviz_current_waypoint_topic", "/current_waypoint");
  this->declare_parameter("rviz_lookahead_waypoint_topic",
                          "/lookahead_waypoint");
  this->declare_parameter("rviz_speed_offset_waypoint_topic",
                          "/speed_offset_waypoint");
  this->declare_parameter("rviz_runtime_params_topic", "/pp_runtime_params");
  this->declare_parameter("rviz_runtime_params_x", 0.0);
  this->declare_parameter("rviz_runtime_params_y", 0.0);
  this->declare_parameter("rviz_runtime_params_z", 1.2);
  this->declare_parameter("rf_topic", "/rf");
  this->declare_parameter("rf_speed_scale_channel", 6);
  this->declare_parameter("rf_max_limit_channel", 7);
  this->declare_parameter("rf_enable_channel", 8);
  this->declare_parameter("rf_enable_threshold", 1500);
  this->declare_parameter("rf_value_min", 1000);
  this->declare_parameter("rf_value_max", 2000);
  // VESC sensors/core 기반 휠 속도
  this->declare_parameter("vesc_state_topic", "sensors/core");
  this->declare_parameter("speed_to_erpm_gain", 3172.47);
  this->declare_parameter("speed_to_erpm_offset", 0.0);
  this->declare_parameter("wheel_speed_deadband", 0.05);
  this->declare_parameter("wheel_speed_scale", 2.6);
  this->declare_parameter("wheel_speed_timeout", 0.5);
  // 런치 스타트
  this->declare_parameter("launch_start_reset_topic", "/launch_start_reset");
  this->declare_parameter("launch_start_enabled", true);
  this->declare_parameter("launch_start_channel", 5);
  this->declare_parameter("launch_start_channel_threshold", 1800);
  this->declare_parameter("launch_start_engage_diff", 1.0);
  this->declare_parameter("launch_start_release_diff", 0.5);
  this->declare_parameter("launch_start_accel", 3.0);
  this->declare_parameter("global_refFrame", "map");
  this->declare_parameter("path_is_circular", true);
  this->declare_parameter("min_lookahead", 0.5);
  this->declare_parameter("max_lookahead", 1.0);
  this->declare_parameter("lookahead_ratio", 8.0);
  this->declare_parameter("speed_profile_distance_offset", 0.0);
  this->declare_parameter("min_searching_idx_offset", 10);
  this->declare_parameter("max_searching_idx_offset", 40);
  this->declare_parameter("K_p", 0.5);
  this->declare_parameter("K_d", 0.1);  // 미분 게인
  this->declare_parameter("K_i", 0.05); // 추가된 적분 게인
  this->declare_parameter("steering_limit", 25.0);
  this->declare_parameter("velocity_percentage", 0.85);
  this->declare_parameter("max_speed_limit_percentage", 1.0);
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
  this->declare_parameter("drive_output_rate_hz", 50.0);
  this->declare_parameter("publish_drive_on_odom", true);
  this->declare_parameter("visualization_rate_hz", 20.0);
  this->declare_parameter("steer_latest_blend", 0.10);
  this->declare_parameter("steer_large_change_blend", 0.55);
  this->declare_parameter("steer_blend_change_threshold_deg", 10.0);
  this->declare_parameter("steer_speed_filter_start_speed", 0.0);
  this->declare_parameter("steer_speed_filter_end_speed", 0.0);
  this->declare_parameter("steer_speed_filter_final_blend", 0.10);
  this->declare_parameter("speed_latest_blend", 0.90);
  this->declare_parameter("slow_with_obs", true);
  this->declare_parameter("obs_slow_th", 3.0);
  this->declare_parameter("obs_slow_percentage", 0.6);

  // 초기화 전용 파라미터 읽어오기 (런타임 변경 불가)
  odom_topic = this->get_parameter("odom_topic").as_string();
  car_refFrame = this->get_parameter("car_refFrame").as_string();
  path_topic = this->get_parameter("path_topic").as_string();
  rviz_current_waypoint_topic =
      this->get_parameter("rviz_current_waypoint_topic").as_string();
  rviz_lookahead_waypoint_topic =
      this->get_parameter("rviz_lookahead_waypoint_topic").as_string();
  rviz_speed_offset_waypoint_topic =
      this->get_parameter("rviz_speed_offset_waypoint_topic").as_string();
  rviz_runtime_params_topic =
      this->get_parameter("rviz_runtime_params_topic").as_string();
  rf_topic = this->get_parameter("rf_topic").as_string();
  vesc_state_topic = this->get_parameter("vesc_state_topic").as_string();
  launch_start_reset_topic =
      this->get_parameter("launch_start_reset_topic").as_string();
  global_refFrame = this->get_parameter("global_refFrame").as_string();
  path_is_circular = this->get_parameter("path_is_circular").as_bool();

  // 런타임 변경 가능 파라미터는 apply_runtime_parameter()로 일괄 반영
  for (const char *name : kRuntimeParameterNames) {
    apply_runtime_parameter(this->get_parameter(name));
  }

  // 초기 적분 오차 초기화
  integral_error = 0.0;
  x_car_world = 0.0;
  y_car_world = 0.0;
  car_heading = 0.0;
  previous_speed_reduction = 0.0;
  target_steer = 0.0;
  target_speed = 0.0;
  output_steer = 0.0;
  output_speed = 0.0;
  current_lookahead_distance = 0.0;
  observed_path_max_speed = 0.0;
  rf_runtime_control_active = false;
  wheel_speed_measured_ = 0.0;
  wheel_speed_valid_ = false;
  launch_start_active_ = false;
  launch_start_pending_ = false;
  launch_start_ramp_speed_ = 0.0;
  launch_start_time_valid_ = false;
  launch_start_prev_raw_ = 0;
  launch_start_prev_raw_valid_ = false;
  has_target_command_ = false;
  output_command_initialized_ = false;
  num_waypoints = 0;

  // 경로 수신 플래그 초기화
  path_received_ = false;

  // Subscriber, Publisher, Timer 등 초기화
  rclcpp::QoS odom_qos(rclcpp::KeepLast(1));
  odom_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
  odom_qos.durability(rclcpp::DurabilityPolicy::Volatile);
  subscription_odom = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, odom_qos, std::bind(&PurePursuit::odom_callback, this, _1));

  subscription_odom_obs = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "/static_obstacle", 10, std::bind(&PurePursuit::obs_odom_callback, this, _1));
  obs_status = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "/obj_flag", 10, std::bind(&PurePursuit::obs_status_callback, this, _1));

  rclcpp::QoS pathQos(rclcpp::KeepLast(10));
  pathQos.reliability(rclcpp::ReliabilityPolicy::Reliable);
  //pathQos.durability(rclcpp::DurabilityPolicy::TransientLocal);

  subscription_path = this->create_subscription<nav_msgs::msg::Path>(
      path_topic, pathQos,
      std::bind(&PurePursuit::path_callback, this, std::placeholders::_1));

  subscription_rf = this->create_subscription<std_msgs::msg::UInt16MultiArray>(
      rf_topic, 10, std::bind(&PurePursuit::rf_callback, this, _1));

  // VESC telemetry는 최신값만 필요하므로 얕은 큐 + BestEffort
  rclcpp::QoS vesc_qos(rclcpp::KeepLast(1));
  vesc_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
  subscription_vesc_state =
      this->create_subscription<vesc_msgs::msg::VescStateStamped>(
          vesc_state_topic, vesc_qos,
          std::bind(&PurePursuit::vesc_state_callback, this, _1));

  subscription_launch_start_reset =
      this->create_subscription<std_msgs::msg::Bool>(
          launch_start_reset_topic, 10,
          std::bind(&PurePursuit::launch_start_reset_callback, this, _1));

  configure_drive_publisher();
  configure_drive_output_timer();
  vis_current_point_pub =
      this->create_publisher<visualization_msgs::msg::Marker>(
          rviz_current_waypoint_topic, 10);
  vis_lookahead_point_pub =
      this->create_publisher<visualization_msgs::msg::Marker>(
          rviz_lookahead_waypoint_topic, 10);
  vis_speed_point_pub =
      this->create_publisher<visualization_msgs::msg::Marker>(
          rviz_speed_offset_waypoint_topic, 10);
  vis_runtime_params_pub =
      this->create_publisher<visualization_msgs::msg::Marker>(
          rviz_runtime_params_topic, 10);

  runtime_param_callback_handle_ = this->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        // 검증: 숫자 파라미터는 유한값이어야 하고,
        // kRuntimeParameterBounds에 등록된 항목은 그 범위 안이어야 합니다.
        // set_parameters_atomically로 들어온 묶음은 하나만 걸려도 전체가 거부됩니다.
        // (일반 set_parameters는 rclcpp가 파라미터별로 호출하므로 개별 판정)
        for (const auto &parameter : parameters) {
          std::string reason;
          if (!validate_runtime_parameter(parameter, &reason)) {
            result.successful = false;
            result.reason = reason;
            RCLCPP_WARN(this->get_logger(), "Rejected parameter update: %s",
                        reason.c_str());
            return result;
          }
        }

        // 반영: 런타임 변경 가능한 파라미터를 멤버 변수에 적용
        bool drive_output_config_changed = false;
        for (const auto &parameter : parameters) {
          if (!apply_runtime_parameter(parameter)) {
            continue;
          }

          const std::string &name = parameter.get_name();
          if (name == "drive_topic" || name == "test_mode" ||
              name == "drive_test_topic" || name == "drive_output_rate_hz") {
            drive_output_config_changed = true;
          }
        }

        if (drive_output_config_changed) {
          configure_drive_publisher();
          configure_drive_output_timer();
        }

        return result;
      });

  runtime_param_visualization_timer_ = this->create_wall_timer(
      200ms, std::bind(&PurePursuit::visualize_runtime_params, this));

  RCLCPP_INFO(this->get_logger(), "Pure pursuit node has been launched");

  // RF 실시간 조절 설정 요약 및 임계값 검증
  {
    const int rf_raw_min = std::min(rf_value_min, rf_value_max);
    const int rf_raw_max = std::max(rf_value_min, rf_value_max);
    RCLCPP_INFO(this->get_logger(),
                "RF runtime control: speed ch[%d], max limit ch[%d], "
                "enable ch[%d] >= %d, raw range [%d, %d]",
                rf_speed_scale_channel, rf_max_limit_channel, rf_enable_channel,
                rf_enable_threshold, rf_raw_min, rf_raw_max);
    RCLCPP_INFO(this->get_logger(),
                "Launch start: %s, trigger ch[%d] rising past %d, "
                "engage >= %.2f m/s, release <= %.2f m/s, accel %.2f m/s^2",
                launch_start_enabled ? "enabled" : "disabled",
                launch_start_channel, launch_start_channel_threshold,
                launch_start_engage_diff, launch_start_release_diff,
                launch_start_accel);
    RCLCPP_INFO(this->get_logger(),
                "Wheel speed: %s, erpm gain %.2f, offset %.2f, "
                "command-domain scale %.2f",
                vesc_state_topic.c_str(), speed_to_erpm_gain,
                speed_to_erpm_offset, wheel_speed_scale);
    if (launch_start_channel == rf_speed_scale_channel ||
        launch_start_channel == rf_max_limit_channel ||
        launch_start_channel == rf_enable_channel) {
      RCLCPP_WARN(this->get_logger(),
                  "launch_start_channel(%d) is already used by another RF "
                  "channel: any rising edge past %d on it also fires launch "
                  "start",
                  launch_start_channel, launch_start_channel_threshold);
    }
    if (launch_start_release_diff >= launch_start_engage_diff) {
      RCLCPP_WARN(this->get_logger(),
                  "launch_start_release_diff(%.2f) >= "
                  "launch_start_engage_diff(%.2f): the latch releases as soon "
                  "as it engages",
                  launch_start_release_diff, launch_start_engage_diff);
    }
    if (rf_enable_threshold <= rf_raw_min) {
      RCLCPP_WARN(this->get_logger(),
                  "rf_enable_threshold(%d) <= raw min(%d): RF runtime control "
                  "stays always enabled",
                  rf_enable_threshold, rf_raw_min);
    } else if (rf_enable_threshold > rf_raw_max) {
      RCLCPP_WARN(this->get_logger(),
                  "rf_enable_threshold(%d) > raw max(%d): RF runtime control "
                  "never gets enabled",
                  rf_enable_threshold, rf_raw_max);
    }
  }

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
                            const double &y1, const double &y2) const {
  return std::hypot(x2 - x1, y2 - y1);
}

double PurePursuit::adjacent_segment_length(int idx, int next_idx,
                                            int direction) const {
  if (num_waypoints <= 1) {
    return 0.0;
  }

  int segment_idx = direction >= 0 ? idx : next_idx;
  if (path_is_circular) {
    segment_idx = path_idx_limiter(segment_idx);
  }

  if (segment_idx >= 0 &&
      segment_idx < static_cast<int>(waypoints.segment_lengths.size())) {
    return waypoints.segment_lengths[segment_idx];
  }

  if (idx >= 0 && idx < num_waypoints && next_idx >= 0 &&
      next_idx < num_waypoints) {
    return p2pdist(waypoints.X[idx], waypoints.X[next_idx], waypoints.Y[idx],
                   waypoints.Y[next_idx]);
  }
  return 0.0;
}

double PurePursuit::current_max_speed_limit() const {
  if (observed_path_max_speed <= 0.0) {
    return -1.0;
  }

  const double limit_ratio =
      std::clamp(max_speed_limit_percentage, 0.0, 1.0);
  return observed_path_max_speed * limit_ratio;
}

double PurePursuit::apply_max_speed_limit(double speed) const {
  double limited_speed = std::max(0.0, speed);
  const double speed_limit = current_max_speed_limit();
  if (speed_limit >= 0.0) {
    limited_speed = std::min(limited_speed, speed_limit);
  }
  return limited_speed;
}

double PurePursuit::rf_raw_to_unit(int raw_value) const {
  const int raw_min = std::min(rf_value_min, rf_value_max);
  const int raw_max = std::max(rf_value_min, rf_value_max);
  const int raw_range = raw_max - raw_min;
  if (raw_range <= 0) {
    return 0.0;
  }

  const double normalized =
      static_cast<double>(raw_value - raw_min) / static_cast<double>(raw_range);
  return std::clamp(normalized, 0.0, 1.0);
}

bool PurePursuit::read_rf_channel(
    const std_msgs::msg::UInt16MultiArray &rf_msg, int channel_index,
    int *raw_value) const {
  if (channel_index < 0 ||
      channel_index >= static_cast<int>(rf_msg.data.size())) {
    return false;
  }

  if (raw_value) {
    *raw_value = static_cast<int>(rf_msg.data[channel_index]);
  }
  return true;
}

void PurePursuit::set_runtime_percentages(double velocity_scale,
                                          double max_speed_limit_scale) {
  velocity_scale = std::clamp(velocity_scale, 0.0, 1.0);
  max_speed_limit_scale = std::clamp(max_speed_limit_scale, 0.0, 1.0);

  constexpr double kUpdateDeadband = 0.001;
  if (std::abs(velocity_scale - velocity_percentage) < kUpdateDeadband &&
      std::abs(max_speed_limit_scale - max_speed_limit_percentage) <
          kUpdateDeadband) {
    return;
  }

  const auto result = this->set_parameters_atomically(
      {rclcpp::Parameter("velocity_percentage", velocity_scale),
       rclcpp::Parameter("max_speed_limit_percentage", max_speed_limit_scale)});

  if (!result.successful) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "Failed to apply RF runtime percentages: %s",
        result.reason.c_str());
    return;
  }

  velocity_percentage = velocity_scale;
  max_speed_limit_percentage = max_speed_limit_scale;
}

std::string PurePursuit::selected_drive_topic() const {
  const std::string selected_topic = test_mode ? drive_test_topic : drive_topic;
  if (!selected_topic.empty()) {
    return selected_topic;
  }
  return test_mode ? "/drive_test" : "/drive";
}

void PurePursuit::configure_drive_publisher() {
  const std::string selected_topic = selected_drive_topic();
  if (publisher_drive && drive_output_topic == selected_topic) {
    return;
  }

  drive_output_topic = selected_topic;
  rclcpp::QoS drive_qos(rclcpp::KeepLast(1));
  drive_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
  drive_qos.durability(rclcpp::DurabilityPolicy::Volatile);
  publisher_drive =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          drive_output_topic, drive_qos);
  RCLCPP_INFO(this->get_logger(), "Drive command output topic: %s%s",
              drive_output_topic.c_str(), test_mode ? " (test mode)" : "");
}

void PurePursuit::configure_drive_output_timer() {
  const double selected_rate = std::clamp(drive_output_rate_hz, 1.0, 200.0);
  if (drive_output_timer_ &&
      std::abs(active_drive_output_rate_hz - selected_rate) <= 1e-6) {
    return;
  }

  active_drive_output_rate_hz = selected_rate;
  const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / active_drive_output_rate_hz));
  drive_output_timer_ = this->create_wall_timer(
      period, std::bind(&PurePursuit::drive_output_timer_callback, this));
  RCLCPP_INFO(this->get_logger(), "Drive command output rate: %.1f Hz",
              active_drive_output_rate_hz);
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
  waypoints.segment_lengths.clear();
  waypoints.path_length = 0.0;
  waypoints.index = 0;
  waypoints.velocity_index = -1;
  waypoints.speed_index = -1;
  has_target_command_ = false;
  output_command_initialized_ = false;

  double received_path_max_speed = 0.0;

  // 새 경로의 포즈를 순회하면서 좌표와 속도(v)를 저장
  for (const auto &pose_stamped : path_msg->poses) {
    const auto &pos = pose_stamped.pose.position;
    waypoints.X.push_back(pos.x);
    waypoints.Y.push_back(pos.y);
    // Path 메시지에서 z 값은 속도 정보를 담고 있다고 가정
    waypoints.V.push_back(pos.z);
    if (std::isfinite(pos.z)) {
      received_path_max_speed =
          std::max(received_path_max_speed, std::max(0.0, pos.z));
    }
  }

  num_waypoints = static_cast<int>(waypoints.X.size());
  waypoints.segment_lengths.assign(num_waypoints, 0.0);
  if (num_waypoints > 1) {
    const int segment_count =
        path_is_circular ? num_waypoints : num_waypoints - 1;
    for (int i = 0; i < segment_count; ++i) {
      const int next_idx = path_is_circular ? path_idx_limiter(i + 1) : i + 1;
      const double segment_length =
          p2pdist(waypoints.X[i], waypoints.X[next_idx], waypoints.Y[i],
                  waypoints.Y[next_idx]);
      waypoints.segment_lengths[i] = segment_length;
      waypoints.path_length += segment_length;
    }
  }

  if (received_path_max_speed > observed_path_max_speed) {
    observed_path_max_speed = received_path_max_speed;
    const double speed_limit = current_max_speed_limit();
    RCLCPP_INFO(this->get_logger(),
                "Observed path max speed updated: %.2fm/s, speed limit: %.2fm/s",
                observed_path_max_speed, speed_limit);
  }

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

void PurePursuit::visualize_speed_point(Eigen::Vector3d &point) {
  auto marker = visualization_msgs::msg::Marker();
  marker.header.frame_id = global_refFrame;
  marker.header.stamp = this->now();
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.3;
  marker.scale.y = 0.3;
  marker.scale.z = 0.3;
  marker.color.a = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 0.8;
  marker.pose.position.x = point(0);
  marker.pose.position.y = point(1);
  marker.id = 1;
  vis_speed_point_pub->publish(marker);
}

void PurePursuit::visualize_runtime_params() {
  if (!vis_runtime_params_pub) {
    return;
  }

  auto marker = visualization_msgs::msg::Marker();
  marker.header.frame_id = global_refFrame.empty() ? "map" : global_refFrame;
  marker.header.stamp = this->now();
  marker.ns = "pure_pursuit_runtime_params";
  marker.id = 1;
  marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = rviz_runtime_params_x;
  marker.pose.position.y = rviz_runtime_params_y;
  marker.pose.position.z = rviz_runtime_params_z;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.35;
  marker.color.a = 1.0;
  marker.color.r = 0.1;
  marker.color.g = 1.0;
  marker.color.b = 0.8;

  std::ostringstream text;
  text << std::fixed << std::setprecision(2)
       << "speed scale: " << velocity_percentage << '\n'
       << "max limit: " << max_speed_limit_percentage << '\n';

  double wheel_speed = 0.0;
  if (command_domain_wheel_speed(&wheel_speed)) {
    text << "wheel: " << wheel_speed;
  } else {
    text << "wheel: n/a";
  }
  if (launch_start_active_) {
    text << '\n' << "LAUNCH " << launch_start_ramp_speed_ << " -> "
         << target_speed;
  }
  marker.text = text.str();

  vis_runtime_params_pub->publish(marker);
}

bool PurePursuit::should_publish_visualization() {
  if (visualization_rate_hz <= 0.0) {
    return false;
  }

  const auto now = this->now();
  const double period_sec = 1.0 / std::max(visualization_rate_hz, 1e-6);
  if (!visualization_time_initialized_) {
    last_visualization_time_ = now;
    visualization_time_initialized_ = true;
    return true;
  }

  const double elapsed_sec = (now - last_visualization_time_).seconds();
  if (elapsed_sec < 0.0 || elapsed_sec >= period_sec) {
    last_visualization_time_ = now;
    return true;
  }

  return false;
}

int PurePursuit::path_idx_limiter(int idx) const {
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

  if (path_is_circular && waypoints.path_length > 1e-6) {
    remaining_distance = std::fmod(remaining_distance, waypoints.path_length);
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
        adjacent_segment_length(idx, next_idx, direction);
    idx = next_idx;
    ++segments_checked;

    if (segment_distance <= 1e-6) {
      continue;
    }
    remaining_distance -= segment_distance;
  }

  return path_idx_limiter(idx);
}

Eigen::Vector3d PurePursuit::sample_path_point_by_distance(
    int start_idx, double distance, int *reached_idx) {
  if (num_waypoints <= 0) {
    if (reached_idx) {
      *reached_idx = 0;
    }
    return Eigen::Vector3d::Zero();
  }

  int idx = path_idx_limiter(start_idx);
  if (num_waypoints == 1 || std::abs(distance) <= 1e-6) {
    if (reached_idx) {
      *reached_idx = idx;
    }
    return Eigen::Vector3d(waypoints.X[idx], waypoints.Y[idx], 0.0);
  }

  const int direction = distance >= 0.0 ? 1 : -1;
  double remaining_distance = std::abs(distance);

  if (path_is_circular && waypoints.path_length > 1e-6) {
    remaining_distance = std::fmod(remaining_distance, waypoints.path_length);
  }

  if (remaining_distance <= 1e-6) {
    if (reached_idx) {
      *reached_idx = idx;
    }
    return Eigen::Vector3d(waypoints.X[idx], waypoints.Y[idx], 0.0);
  }

  const int max_segments = path_is_circular ? num_waypoints : num_waypoints - 1;
  int segments_checked = 0;
  while (segments_checked < max_segments) {
    const int next_idx =
        path_is_circular ? path_idx_limiter(idx + direction) : idx + direction;
    if (!path_is_circular && (next_idx < 0 || next_idx >= num_waypoints)) {
      break;
    }

    const double segment_distance =
        adjacent_segment_length(idx, next_idx, direction);
    ++segments_checked;

    if (segment_distance <= 1e-6) {
      idx = next_idx;
      continue;
    }

    if (remaining_distance <= segment_distance) {
      const double ratio = remaining_distance / segment_distance;
      if (reached_idx) {
        *reached_idx = next_idx;
      }
      return Eigen::Vector3d(
          waypoints.X[idx] + ratio * (waypoints.X[next_idx] - waypoints.X[idx]),
          waypoints.Y[idx] + ratio * (waypoints.Y[next_idx] - waypoints.Y[idx]),
          0.0);
    }

    remaining_distance -= segment_distance;
    idx = next_idx;
  }

  if (reached_idx) {
    *reached_idx = path_idx_limiter(idx);
  }
  return Eigen::Vector3d(waypoints.X[path_idx_limiter(idx)],
                         waypoints.Y[path_idx_limiter(idx)], 0.0);
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
  current_lookahead_distance = std::min(
      std::max(min_lookahead, max_lookahead * curr_velocity / lookahead_ratio),
      max_lookahead);

  // lookahead point는 waypoint 자체가 아니라 segment 위의 정확한 거리 지점을 사용합니다.
  int reached_idx = waypoints.velocity_index;
  lookahead_point_world = sample_path_point_by_distance(
      waypoints.velocity_index, current_lookahead_distance, &reached_idx);
  waypoints.index = reached_idx;

  waypoints.speed_index =
      advance_index_by_distance(waypoints.velocity_index,
                                speed_profile_distance_offset);
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
  int speed_idx =
      waypoints.speed_index >= 0 ? path_idx_limiter(waypoints.speed_index)
                                 : vel_idx;
  // lookahead_point_world는 get_waypoint_new()에서 segment 보간으로 갱신됩니다.
  if (look_idx < 0 || look_idx >= num_waypoints) {
    lookahead_point_world << waypoints.X[vel_idx], waypoints.Y[vel_idx], 0.0;
  }
  current_point_world << waypoints.X[vel_idx], waypoints.Y[vel_idx], 0.0;
  speed_point_world << waypoints.X[speed_idx], waypoints.Y[speed_idx], 0.0;

  if (should_publish_visualization()) {
    visualize_lookahead_point(lookahead_point_world);
    visualize_current_point(current_point_world);
    visualize_speed_point(speed_point_world);
  }

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

void PurePursuit::update_target_command(double steering_angle) {
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
  target_steer =
      std::clamp(expo_steering, -steering_limit_rad, steering_limit_rad);
  const double expo_scale =
      std::abs(adjusted_steering) > 1e-6
          ? std::abs(target_steer) / std::abs(adjusted_steering)
          : 1.0;

  const double speed_before_max_limit = desired_speed;
  target_speed = apply_max_speed_limit(desired_speed);
  previous_speed_reduction =
      std::max(0.0, base_desired_speed - speed_before_max_limit);
  has_target_command_ = true;

  RCLCPP_DEBUG_THROTTLE(
      this->get_logger(), *this->get_clock(), 200,
      "index: %d ... distance: %.2fm ... TargetSpeed: %.2fm/s ... OutputSpeed: %.2fm/s ... Steering "
      "Angle: %.2f ... Raw Steering: %.2f ... SpeedIdx: %d ... "
      "SpeedOffset: %.2fm ... "
      "MaxPathSpeed: %.2f ... MaxSpeedCap: %.2f ... "
      "Steer Scale: %.2f ... "
      "Expo Scale: %.2f ... "
      "SpeedAdj: %.2f ... PrevReduction: %.2f ... K_p: %.2f "
      "... K_i: %.2f ... velocity_percentage: %.2f",
      waypoints.index,
      p2pdist(lookahead_point_world(0), x_car_world,
              lookahead_point_world(1), y_car_world),
      target_speed, output_speed, to_degrees(target_steer),
      to_degrees(raw_steering), waypoints.speed_index,
      speed_profile_distance_offset, observed_path_max_speed,
      current_max_speed_limit(), steer_scale, expo_scale, total_speed_adjust,
      previous_speed_reduction, K_p, K_i, velocity_percentage);
}

void PurePursuit::drive_output_timer_callback() {
  if (!has_target_command_ || !publisher_drive) {
    return;
  }

  const double speed_alpha = std::clamp(speed_latest_blend, 0.0, 1.0);

  if (!output_command_initialized_) {
    output_steer = target_steer;
    output_speed = target_speed;
    output_command_initialized_ = true;
  } else {
    const double steer_low_alpha = std::clamp(steer_latest_blend, 0.0, 1.0);
    const double steer_high_alpha =
        std::clamp(steer_large_change_blend, steer_low_alpha, 1.0);
    const double steer_threshold_rad =
        std::max(to_radians(steer_blend_change_threshold_deg), 1e-6);
    const double steer_delta_abs = std::abs(target_steer - output_steer);
    const double steer_blend_ratio =
        std::clamp(steer_delta_abs / steer_threshold_rad, 0.0, 1.0);
    const double base_steer_alpha =
        steer_low_alpha +
        steer_blend_ratio * (steer_high_alpha - steer_low_alpha);
    double steer_alpha = base_steer_alpha;

    const double command_speed_abs = std::abs(target_speed);
    const double speed_start =
        std::max(0.0, steer_speed_filter_start_speed);
    const double speed_end = std::max(0.0, steer_speed_filter_end_speed);
    double speed_blend_ratio = 0.0;
    if (speed_end > speed_start) {
      speed_blend_ratio =
          std::clamp((command_speed_abs - speed_start) /
                         (speed_end - speed_start),
                     0.0, 1.0);
    } else if (speed_start > 0.0 && command_speed_abs >= speed_start) {
      speed_blend_ratio = 1.0;
    }

    if (speed_blend_ratio > 0.0) {
      const double speed_final_alpha =
          std::clamp(steer_speed_filter_final_blend, steer_alpha, 1.0);
      steer_alpha += speed_blend_ratio * (speed_final_alpha - steer_alpha);
    }

    output_steer += steer_alpha * (target_steer - output_steer);
    output_speed += speed_alpha * (target_speed - output_speed);
  }

  // 런치 스타트 래치가 걸린 동안에는 위 블렌딩 대신 설정 가속도 램프를
  // 그대로 내보냅니다. 래치가 풀리면 램프 값에서 이어서 기존 동작으로
  // 돌아가므로 전환 지점에 단차가 생기지 않습니다.
  update_launch_start();
  if (launch_start_active_) {
    output_speed = launch_start_ramp_speed_;
  }

  const double steering_limit_rad = to_radians(steering_limit);
  output_steer =
      std::clamp(output_steer, -steering_limit_rad, steering_limit_rad);
  output_speed = apply_max_speed_limit(output_speed);
  curr_velocity = output_speed;

  auto drive_msgObj = ackermann_msgs::msg::AckermannDriveStamped();
  drive_msgObj.header.stamp = this->now();
  drive_msgObj.drive.steering_angle = output_steer;
  drive_msgObj.drive.speed = output_speed;

  publisher_drive->publish(drive_msgObj);
}

void PurePursuit::odom_callback(
    const nav_msgs::msg::Odometry::ConstSharedPtr odom_submsgObj) {
  x_car_world = odom_submsgObj->pose.pose.position.x;
  y_car_world = odom_submsgObj->pose.pose.position.y;
  const auto &orientation = odom_submsgObj->pose.pose.orientation;
  tf2::Quaternion q(orientation.x, orientation.y, orientation.z,
                    orientation.w);
  car_heading = tf2::getYaw(q);

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
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "No path received yet, waiting for path... (current odom will be ignored)");
    return;
  }

  // waypoint 업데이트, odom 기준 좌표 변환, 그리고 PID 제어를 통한 steering 계산
  get_waypoint_new();
  transformandinterp_waypoint();
  double steering_angle = p_controller();
  update_target_command(steering_angle);
  if (publish_drive_on_odom) {
    drive_output_timer_callback();
  }
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

void PurePursuit::rf_callback(
    const std_msgs::msg::UInt16MultiArray::ConstSharedPtr rf_msg) {
  if (!rf_msg) {
    return;
  }

  // 런치 스타트 트리거는 enable 스위치와 무관하게 판정합니다.
  // 임계값 이하였다가 초과로 "올라가는" 순간에만 한 번 발생하고,
  // 첫 메시지는 직전 값이 없으므로 엣지로 보지 않습니다.
  int launch_raw = 0;
  if (read_rf_channel(*rf_msg, launch_start_channel, &launch_raw)) {
    const bool is_above = launch_raw > launch_start_channel_threshold;
    const bool was_above =
        launch_start_prev_raw_ > launch_start_channel_threshold;
    if (launch_start_prev_raw_valid_ && is_above && !was_above) {
      request_launch_start("RF channel");
    }
    launch_start_prev_raw_ = launch_raw;
    launch_start_prev_raw_valid_ = true;
  }

  int enable_raw = 0;
  if (!read_rf_channel(*rf_msg, rf_enable_channel, &enable_raw)) {
    rf_runtime_control_active = false;
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "RF enable channel index %d is unavailable. /rf size: %zu",
        rf_enable_channel, rf_msg->data.size());
    return;
  }

  const bool should_apply_rf = enable_raw >= rf_enable_threshold;
  if (!should_apply_rf) {
    if (rf_runtime_control_active) {
      RCLCPP_INFO(this->get_logger(),
                  "RF runtime control disabled by channel index %d: %d < %d",
                  rf_enable_channel, enable_raw, rf_enable_threshold);
    }
    rf_runtime_control_active = false;
    return;
  }

  int speed_scale_raw = 0;
  int max_limit_raw = 0;
  if (!read_rf_channel(*rf_msg, rf_speed_scale_channel, &speed_scale_raw) ||
      !read_rf_channel(*rf_msg, rf_max_limit_channel, &max_limit_raw)) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "RF parameter channel is unavailable. speed index: %d, max index: %d, "
        "/rf size: %zu",
        rf_speed_scale_channel, rf_max_limit_channel, rf_msg->data.size());
    return;
  }

  if (!rf_runtime_control_active) {
    RCLCPP_INFO(this->get_logger(),
                "RF runtime control enabled by channel index %d: %d >= %d",
                rf_enable_channel, enable_raw, rf_enable_threshold);
  }
  rf_runtime_control_active = true;

  set_runtime_percentages(rf_raw_to_unit(speed_scale_raw),
                          rf_raw_to_unit(max_limit_raw));
}

// VESC telemetry(sensors/core)의 전기 RPM을 휠 속도(m/s)로 변환해 보관합니다.
//   speed = (erpm - speed_to_erpm_offset) / speed_to_erpm_gain
// vesc_to_odom과 동일한 식이며 저속 데드밴드도 동일하게 적용합니다.
void PurePursuit::vesc_state_callback(
    const vesc_msgs::msg::VescStateStamped::ConstSharedPtr state_msg) {
  if (!state_msg) {
    return;
  }

  if (std::abs(speed_to_erpm_gain) < 1e-6) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "speed_to_erpm_gain(%.6f) is too small to convert "
                         "wheel speed",
                         speed_to_erpm_gain);
    return;
  }

  double speed =
      (state_msg->state.speed - speed_to_erpm_offset) / speed_to_erpm_gain;
  if (std::abs(speed) < std::max(0.0, wheel_speed_deadband)) {
    speed = 0.0;
  }

  wheel_speed_measured_ = speed;
  wheel_speed_stamp_ = this->now();
  wheel_speed_valid_ = true;
}

// 다른 노드가 런치 스타트를 걸거나(true) 해제(false)합니다.
void PurePursuit::launch_start_reset_callback(
    const std_msgs::msg::Bool::ConstSharedPtr reset_msg) {
  if (!reset_msg) {
    return;
  }

  if (reset_msg->data) {
    request_launch_start("reset topic");
  } else {
    cancel_launch_start("reset topic");
  }
}

// 측정 휠 속도를 명령 속도(path speed) 도메인으로 변환해 돌려줍니다.
// 값이 없거나 오래됐으면 false.
bool PurePursuit::command_domain_wheel_speed(double *speed) const {
  if (!wheel_speed_valid_) {
    return false;
  }

  if (wheel_speed_timeout > 0.0) {
    const double age = (this->now() - wheel_speed_stamp_).seconds();
    if (age < 0.0 || age > wheel_speed_timeout) {
      return false;
    }
  }

  if (speed) {
    *speed = wheel_speed_measured_ * wheel_speed_scale;
  }
  return true;
}

// 런치 스타트를 요청합니다. 실제 래치 여부는 목표 속도를 알 수 있는
// update_launch_start()에서 판정하므로 여기서는 대기 상태로만 둡니다.
// 이미 진행 중이어도 현재 휠 속도 기준으로 다시 시작합니다.
void PurePursuit::request_launch_start(const char *source) {
  if (!launch_start_enabled) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "Launch start request from %s ignored: "
                         "launch_start_enabled is false",
                         source);
    return;
  }

  launch_start_pending_ = true;
  launch_start_active_ = false;
  launch_start_time_valid_ = false;
  RCLCPP_INFO(this->get_logger(), "Launch start requested by %s", source);
}

void PurePursuit::cancel_launch_start(const char *source) {
  if (!launch_start_active_ && !launch_start_pending_) {
    return;
  }

  launch_start_active_ = false;
  launch_start_pending_ = false;
  launch_start_time_valid_ = false;
  RCLCPP_INFO(this->get_logger(), "Launch start cancelled by %s", source);
}

// 래치 상태를 갱신합니다. 래치가 걸려 있는 동안 launch_start_ramp_speed_는
// 설정한 가속도만큼만 목표 속도를 따라갑니다. 목표 속도는 차량 위치에 따라
// 계속 바뀌므로, 램프가 그 목표에 release_diff 이내로 붙으면 래치를 풉니다.
void PurePursuit::update_launch_start() {
  if (!launch_start_enabled) {
    launch_start_active_ = false;
    launch_start_pending_ = false;
    return;
  }

  if (launch_start_pending_) {
    launch_start_pending_ = false;

    double wheel_speed = 0.0;
    if (!command_domain_wheel_speed(&wheel_speed)) {
      RCLCPP_WARN(this->get_logger(),
                  "Launch start skipped: no fresh wheel speed on %s",
                  vesc_state_topic.c_str());
    } else if (std::abs(target_speed - wheel_speed) <
               launch_start_engage_diff) {
      // 이미 목표 속도에 붙어 있으면 기존 동작 그대로 둡니다.
      RCLCPP_INFO(this->get_logger(),
                  "Launch start skipped: |target %.2f - wheel %.2f| < %.2f",
                  target_speed, wheel_speed, launch_start_engage_diff);
    } else {
      launch_start_active_ = true;
      launch_start_ramp_speed_ = wheel_speed;
      launch_start_time_valid_ = false;
      RCLCPP_INFO(this->get_logger(),
                  "Launch start engaged: target %.2f m/s, wheel %.2f m/s, "
                  "accel %.2f m/s^2",
                  target_speed, wheel_speed, launch_start_accel);
    }
  }

  if (!launch_start_active_) {
    return;
  }

  const rclcpp::Time now = this->now();
  double dt = 0.0;
  if (launch_start_time_valid_) {
    dt = (now - launch_start_prev_time_).seconds();
  }
  launch_start_prev_time_ = now;
  launch_start_time_valid_ = true;
  // 콜백이 밀리거나 시계가 되감긴 경우 한 번에 크게 튀지 않도록 제한
  dt = std::clamp(dt, 0.0, 0.2);

  const double step = std::max(0.0, launch_start_accel) * dt;
  const double diff = target_speed - launch_start_ramp_speed_;
  if (std::abs(diff) <= step) {
    launch_start_ramp_speed_ = target_speed;
  } else {
    launch_start_ramp_speed_ += std::copysign(step, diff);
  }

  if (std::abs(target_speed - launch_start_ramp_speed_) <=
      launch_start_release_diff) {
    launch_start_active_ = false;
    launch_start_time_valid_ = false;
    RCLCPP_INFO(this->get_logger(),
                "Launch start released: ramp %.2f m/s, target %.2f m/s",
                launch_start_ramp_speed_, target_speed);
  }
}

// 런타임 파라미터 하나가 반영해도 안전한 값인지 검사합니다.
// 숫자가 아닌 파라미터(토픽 이름, 플래그 등)는 검사 없이 통과시킵니다.
bool PurePursuit::validate_runtime_parameter(const rclcpp::Parameter &parameter,
                                             std::string *reason) const {
  const std::string &name = parameter.get_name();

  const RuntimeParameterBound *bound = nullptr;
  for (const auto &candidate : kRuntimeParameterBounds) {
    if (name == candidate.name) {
      bound = &candidate;
      break;
    }
  }

  const auto type = parameter.get_type();
  if (type != rclcpp::ParameterType::PARAMETER_DOUBLE &&
      type != rclcpp::ParameterType::PARAMETER_INTEGER) {
    if (bound != nullptr) {
      if (reason) {
        *reason = name + " must be a number";
      }
      return false;
    }
    return true;
  }

  const double value =
      type == rclcpp::ParameterType::PARAMETER_INTEGER
          ? static_cast<double>(parameter.as_int())
          : parameter.as_double();

  // NaN/inf는 한 번 들어오면 제어 출력과 적분항까지 오염되어 복구되지 않습니다.
  if (!std::isfinite(value)) {
    if (reason) {
      *reason = name + " must be a finite number (NaN/inf is rejected)";
    }
    return false;
  }

  if (bound != nullptr &&
      (value < bound->min_value || value > bound->max_value)) {
    if (reason) {
      std::ostringstream oss;
      oss << name << " must be in [" << bound->min_value << ", ";
      if (bound->max_value == kUnbounded) {
        oss << "inf";
      } else {
        oss << bound->max_value;
      }
      oss << "] (got " << value << ")";
      *reason = oss.str();
    }
    return false;
  }

  return true;
}

// 런타임에 변경 가능한 파라미터 하나를 대응하는 멤버 변수에 반영합니다.
// 처리한 파라미터면 true, 목록에 없는 이름이면 false를 반환합니다.
bool PurePursuit::apply_runtime_parameter(const rclcpp::Parameter &parameter) {
  const std::string &name = parameter.get_name();

  // 속도 비율 파라미터는 int로도 들어올 수 있어 별도 처리
  auto as_clamped_ratio = [&parameter]() {
    const double value =
        parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER
            ? static_cast<double>(parameter.as_int())
            : parameter.as_double();
    return std::clamp(value, 0.0, 1.0);
  };

  if (name == "drive_topic") {
    drive_topic = parameter.as_string();
  } else if (name == "test_mode") {
    test_mode = parameter.as_bool();
  } else if (name == "drive_test_topic") {
    drive_test_topic = parameter.as_string();
  } else if (name == "rviz_runtime_params_x") {
    rviz_runtime_params_x = parameter.as_double();
  } else if (name == "rviz_runtime_params_y") {
    rviz_runtime_params_y = parameter.as_double();
  } else if (name == "rviz_runtime_params_z") {
    rviz_runtime_params_z = parameter.as_double();
  } else if (name == "rf_speed_scale_channel") {
    rf_speed_scale_channel = parameter.as_int();
  } else if (name == "rf_max_limit_channel") {
    rf_max_limit_channel = parameter.as_int();
  } else if (name == "rf_enable_channel") {
    rf_enable_channel = parameter.as_int();
  } else if (name == "rf_enable_threshold") {
    rf_enable_threshold = parameter.as_int();
  } else if (name == "rf_value_min") {
    rf_value_min = parameter.as_int();
  } else if (name == "rf_value_max") {
    rf_value_max = parameter.as_int();
  } else if (name == "K_p") {
    K_p = parameter.as_double();
  } else if (name == "K_d") {
    K_d = parameter.as_double();
  } else if (name == "K_i") {
    K_i = parameter.as_double();
  } else if (name == "heading_error_gain") {
    heading_error_gain = parameter.as_double();
  } else if (name == "velocity_percentage") {
    velocity_percentage = as_clamped_ratio();
  } else if (name == "max_speed_limit_percentage") {
    max_speed_limit_percentage = as_clamped_ratio();
  } else if (name == "min_lookahead") {
    min_lookahead = parameter.as_double();
  } else if (name == "max_lookahead") {
    max_lookahead = parameter.as_double();
  } else if (name == "lookahead_ratio") {
    lookahead_ratio = parameter.as_double();
  } else if (name == "speed_profile_distance_offset") {
    speed_profile_distance_offset = parameter.as_double();
  } else if (name == "steering_limit") {
    steering_limit = parameter.as_double();
  } else if (name == "steering_expo_gain") {
    steering_expo_gain = parameter.as_double();
  } else if (name == "steering_expo_curve") {
    steering_expo_curve = parameter.as_double();
  } else if (name == "steer_reduction_speed_threshold") {
    steer_reduction_speed_threshold = parameter.as_double();
  } else if (name == "steer_reduction_constant_coef") {
    steer_reduction_constant_coef = parameter.as_double();
  } else if (name == "steer_reduction_linear_coef") {
    steer_reduction_linear_coef = parameter.as_double();
  } else if (name == "steer_reduction_min_scale") {
    steer_reduction_min_scale = parameter.as_double();
  } else if (name == "speed_reduction_steer_angle_deg") {
    speed_reduction_angle_threshold = to_radians(parameter.as_double());
  } else if (name == "max_allowed_steer_drop_deg") {
    max_allowed_steer_drop = to_radians(parameter.as_double());
  } else if (name == "speed_reduction_adjust") {
    speed_reduction_adjust = parameter.as_double();
  } else if (name == "speed_reduction_prev_scale") {
    speed_reduction_prev_scale = parameter.as_double();
  } else if (name == "drive_output_rate_hz") {
    drive_output_rate_hz = parameter.as_double();
  } else if (name == "publish_drive_on_odom") {
    publish_drive_on_odom = parameter.as_bool();
  } else if (name == "visualization_rate_hz") {
    visualization_rate_hz = parameter.as_double();
  } else if (name == "steer_latest_blend") {
    steer_latest_blend = parameter.as_double();
  } else if (name == "steer_large_change_blend") {
    steer_large_change_blend = parameter.as_double();
  } else if (name == "steer_blend_change_threshold_deg") {
    steer_blend_change_threshold_deg = parameter.as_double();
  } else if (name == "steer_speed_filter_start_speed") {
    steer_speed_filter_start_speed = parameter.as_double();
  } else if (name == "steer_speed_filter_end_speed") {
    steer_speed_filter_end_speed = parameter.as_double();
  } else if (name == "steer_speed_filter_final_blend") {
    steer_speed_filter_final_blend = parameter.as_double();
  } else if (name == "speed_latest_blend") {
    speed_latest_blend = parameter.as_double();
  } else if (name == "min_searching_idx_offset") {
    min_searching_idx_offset = parameter.as_int();
  } else if (name == "max_searching_idx_offset") {
    max_searching_idx_offset = parameter.as_int();
  } else if (name == "slow_with_obs") {
    slow_with_obs = parameter.as_bool();
  } else if (name == "obs_slow_th") {
    slow_th_dist = parameter.as_double();
  } else if (name == "obs_slow_percentage") {
    slow_amount = parameter.as_double();
  } else if (name == "speed_to_erpm_gain") {
    speed_to_erpm_gain = parameter.as_double();
  } else if (name == "speed_to_erpm_offset") {
    speed_to_erpm_offset = parameter.as_double();
  } else if (name == "wheel_speed_deadband") {
    wheel_speed_deadband = parameter.as_double();
  } else if (name == "wheel_speed_scale") {
    wheel_speed_scale = parameter.as_double();
  } else if (name == "wheel_speed_timeout") {
    wheel_speed_timeout = parameter.as_double();
  } else if (name == "launch_start_enabled") {
    launch_start_enabled = parameter.as_bool();
  } else if (name == "launch_start_channel") {
    launch_start_channel = parameter.as_int();
  } else if (name == "launch_start_channel_threshold") {
    launch_start_channel_threshold = parameter.as_int();
  } else if (name == "launch_start_engage_diff") {
    launch_start_engage_diff = parameter.as_double();
  } else if (name == "launch_start_release_diff") {
    launch_start_release_diff = parameter.as_double();
  } else if (name == "launch_start_accel") {
    launch_start_accel = parameter.as_double();
  } else {
    return false;
  }

  return true;
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
