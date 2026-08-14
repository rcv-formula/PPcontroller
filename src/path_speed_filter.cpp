#include "path_speed_filter.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "rcl_interfaces/msg/set_parameters_result.hpp"

using std::placeholders::_1;

namespace {
// 런타임(ros2 param set)에 변경 가능한 파라미터 목록.
// apply_runtime_parameter()의 분기와 1:1로 대응합니다.
constexpr const char *kRuntimeParameterNames[] = {
    "path_is_circular",
    "decel_preview_distance",
    "decel_drop_gain",
    "activation_speed",
    "activation_ramp_range",
    "max_speed_drop",
};

constexpr double kMinSegmentLength = 1e-6;
} // namespace

PathSpeedFilter::PathSpeedFilter() : Node("path_speed_filter") {
  // 파라미터 선언
  this->declare_parameter("input_path_topic", "/Path");
  this->declare_parameter("output_path_topic", "/Path_filtered");
  this->declare_parameter("path_is_circular", true);
  this->declare_parameter("decel_preview_distance", 1.5);
  this->declare_parameter("decel_drop_gain", 0.5);
  this->declare_parameter("activation_speed", 4.0);
  this->declare_parameter("activation_ramp_range", 1.0);
  this->declare_parameter("max_speed_drop", 3.0);

  // 초기화 전용 파라미터 읽어오기
  input_path_topic_ = this->get_parameter("input_path_topic").as_string();
  output_path_topic_ = this->get_parameter("output_path_topic").as_string();

  // 런타임 변경 가능 파라미터는 apply_runtime_parameter()로 일괄 반영
  for (const char *name : kRuntimeParameterNames) {
    apply_runtime_parameter(this->get_parameter(name));
  }

  // pure_pursuit의 path subscriber와 동일한 QoS 사용
  rclcpp::QoS path_qos(rclcpp::KeepLast(10));
  path_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);

  subscription_path_ = this->create_subscription<nav_msgs::msg::Path>(
      input_path_topic_, path_qos,
      std::bind(&PathSpeedFilter::path_callback, this, _1));
  publisher_path_ =
      this->create_publisher<nav_msgs::msg::Path>(output_path_topic_,
                                                  path_qos);

  // 파라미터 변경 시 마지막 path를 재처리해 즉시 재발행 (런타임 튜닝용)
  param_callback_handle_ = this->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        bool changed = false;
        for (const auto &parameter : parameters) {
          changed = apply_runtime_parameter(parameter) || changed;
        }
        if (changed) {
          publish_filtered_path();
        }
        return result;
      });

  RCLCPP_INFO(this->get_logger(),
              "Path speed filter launched: %s -> %s (preview: %.2fm, "
              "drop gain: %.2f, activation: %.1fm/s)",
              input_path_topic_.c_str(), output_path_topic_.c_str(),
              decel_preview_distance_, decel_drop_gain_, activation_speed_);
}

// 런타임에 변경 가능한 파라미터 하나를 대응하는 멤버 변수에 반영합니다.
// 처리한 파라미터면 true, 목록에 없는 이름이면 false를 반환합니다.
bool PathSpeedFilter::apply_runtime_parameter(
    const rclcpp::Parameter &parameter) {
  const std::string &name = parameter.get_name();

  if (name == "path_is_circular") {
    path_is_circular_ = parameter.as_bool();
  } else if (name == "decel_preview_distance") {
    decel_preview_distance_ = std::max(0.0, parameter.as_double());
  } else if (name == "decel_drop_gain") {
    decel_drop_gain_ = std::max(0.0, parameter.as_double());
  } else if (name == "activation_speed") {
    activation_speed_ = std::max(0.0, parameter.as_double());
  } else if (name == "activation_ramp_range") {
    activation_ramp_range_ = std::max(0.0, parameter.as_double());
  } else if (name == "max_speed_drop") {
    max_speed_drop_ = std::max(0.0, parameter.as_double());
  } else {
    return false;
  }

  return true;
}

void PathSpeedFilter::path_callback(
    const nav_msgs::msg::Path::ConstSharedPtr path_msg) {
  if (!path_msg || path_msg->poses.empty()) {
    RCLCPP_WARN(this->get_logger(),
                "Received empty path message, ignoring it.");
    return;
  }

  last_path_ = path_msg;
  publish_filtered_path();
}

bool PathSpeedFilter::speed_at_distance_ahead(
    const std::vector<double> &speeds,
    const std::vector<double> &segment_lengths, int start_idx,
    double preview_distance, double *speed_ahead, double *traveled) const {
  const int num_waypoints = static_cast<int>(speeds.size());
  double remaining = preview_distance;
  double walked = 0.0;
  int idx = start_idx;
  double result_speed = speeds[idx];

  // 세그먼트를 따라 전방으로 진행. circular면 wrap, 아니면 경로 끝에서 멈춤
  const int max_segments =
      path_is_circular_ ? num_waypoints : num_waypoints - 1 - start_idx;
  for (int step = 0; step < max_segments && remaining > kMinSegmentLength;
       ++step) {
    const int next_idx = path_is_circular_ ? (idx + 1) % num_waypoints
                                           : idx + 1;
    if (!std::isfinite(speeds[next_idx])) {
      return false;
    }

    const double segment_length = segment_lengths[idx];
    if (segment_length <= kMinSegmentLength) {
      idx = next_idx;
      continue;
    }

    if (remaining <= segment_length) {
      // 세그먼트 내부 지점: 호 길이 비율로 속도 선형보간
      const double ratio = remaining / segment_length;
      result_speed =
          speeds[idx] + ratio * (speeds[next_idx] - speeds[idx]);
      walked += remaining;
      remaining = 0.0;
      break;
    }

    remaining -= segment_length;
    walked += segment_length;
    idx = next_idx;
    result_speed = speeds[next_idx];
  }

  *speed_ahead = result_speed;
  *traveled = walked;
  return true;
}

void PathSpeedFilter::publish_filtered_path() {
  if (!last_path_) {
    return;
  }

  nav_msgs::msg::Path out_msg = *last_path_;
  const int num_waypoints = static_cast<int>(out_msg.poses.size());
  if (num_waypoints < 2) {
    publisher_path_->publish(out_msg);
    return;
  }

  // 원본 속도 프로파일과 세그먼트 길이 계산
  // (보정은 항상 원본 프로파일을 기준으로 계산해 순서 의존성을 없앰)
  std::vector<double> speeds(num_waypoints);
  std::vector<double> segment_lengths(num_waypoints, 0.0);
  for (int i = 0; i < num_waypoints; ++i) {
    speeds[i] = out_msg.poses[i].pose.position.z;
  }
  const int segment_count =
      path_is_circular_ ? num_waypoints : num_waypoints - 1;
  for (int i = 0; i < segment_count; ++i) {
    const int next_idx = (i + 1) % num_waypoints;
    const auto &p1 = out_msg.poses[i].pose.position;
    const auto &p2 = out_msg.poses[next_idx].pose.position;
    segment_lengths[i] = std::hypot(p2.x - p1.x, p2.y - p1.y);
  }

  const double preview =
      std::max(decel_preview_distance_, kMinSegmentLength);

  for (int i = 0; i < num_waypoints; ++i) {
    const double v = speeds[i];
    if (!std::isfinite(v) || v <= 0.0) {
      continue;
    }

    double v_ahead = v;
    double traveled = 0.0;
    if (!speed_at_distance_ahead(speeds, segment_lengths, i, preview,
                                 &v_ahead, &traveled) ||
        traveled <= kMinSegmentLength) {
      continue;
    }

    // 등가속 공식 v_ahead^2 = v^2 + 2*a*s 에서 전방 구간의 감가속도 계산.
    // a < 0 이면 앞에 감속 구간이 있다는 뜻
    const double accel =
        (v_ahead * v_ahead - v * v) / (2.0 * traveled);
    const double decel = std::max(0.0, -accel);
    if (decel <= 0.0) {
      continue;
    }

    // activation_speed 경계에서 점프하지 않도록 ramp 구간에 걸쳐 0->1 블렌딩
    const double activation_ratio =
        activation_ramp_range_ > kMinSegmentLength
            ? std::clamp((v - activation_speed_) / activation_ramp_range_,
                         0.0, 1.0)
            : (v >= activation_speed_ ? 1.0 : 0.0);
    if (activation_ratio <= 0.0) {
      continue;
    }

    const double drop =
        std::min(decel_drop_gain_ * decel * activation_ratio,
                 max_speed_drop_);
    out_msg.poses[i].pose.position.z = std::max(0.0, v - drop);
  }

  publisher_path_->publish(out_msg);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node_ptr = std::make_shared<PathSpeedFilter>();
  rclcpp::spin(node_ptr);
  rclcpp::shutdown();
  return 0;
}
