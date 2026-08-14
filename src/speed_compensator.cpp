#include "speed_compensator.hpp"

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
    "speed_to_erpm_gain",
    "speed_to_erpm_offset",
    "activation_speed",
    "activation_ramp_range",
    "decel_boost_gain",
    "accel_boost_gain",
    "decel_boost_max",
    "accel_boost_max",
    "speed_error_deadband",
    "speed_timeout_sec",
};
} // namespace

SpeedCompensator::SpeedCompensator() : Node("speed_compensator") {
  // 파라미터 선언
  this->declare_parameter("input_drive_topic", "/drive_main");
  this->declare_parameter("output_drive_topic", "/drive_filtered");
  this->declare_parameter("speed_source_type", "vesc");
  this->declare_parameter("vesc_state_topic", "/sensors/core");
  this->declare_parameter("odom_topic", "/odom");
  this->declare_parameter("speed_to_erpm_gain", 3172.47);
  this->declare_parameter("speed_to_erpm_offset", 0.0);
  this->declare_parameter("activation_speed", 4.0);
  this->declare_parameter("activation_ramp_range", 1.0);
  this->declare_parameter("decel_boost_gain", 0.0);
  this->declare_parameter("accel_boost_gain", 0.0);
  this->declare_parameter("decel_boost_max", 3.0);
  this->declare_parameter("accel_boost_max", 2.0);
  this->declare_parameter("speed_error_deadband", 0.1);
  this->declare_parameter("speed_timeout_sec", 0.5);

  // 초기화 전용 파라미터 읽어오기
  input_drive_topic_ = this->get_parameter("input_drive_topic").as_string();
  output_drive_topic_ = this->get_parameter("output_drive_topic").as_string();
  speed_source_type_ = this->get_parameter("speed_source_type").as_string();
  vesc_state_topic_ = this->get_parameter("vesc_state_topic").as_string();
  odom_topic_ = this->get_parameter("odom_topic").as_string();

  // 런타임 변경 가능 파라미터는 apply_runtime_parameter()로 일괄 반영
  for (const char *name : kRuntimeParameterNames) {
    apply_runtime_parameter(this->get_parameter(name));
  }

  last_speed_time_ = this->now();

  // drive 입출력은 pure_pursuit의 drive publisher와 동일한 QoS 사용
  rclcpp::QoS drive_qos(rclcpp::KeepLast(1));
  drive_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
  drive_qos.durability(rclcpp::DurabilityPolicy::Volatile);

  subscription_drive_ =
      this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
          input_drive_topic_, drive_qos,
          std::bind(&SpeedCompensator::drive_callback, this, _1));
  publisher_drive_ =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          output_drive_topic_, drive_qos);

  if (speed_source_type_ == "odom") {
    rclcpp::QoS odom_qos(rclcpp::KeepLast(1));
    odom_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
    subscription_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, odom_qos,
        std::bind(&SpeedCompensator::odom_callback, this, _1));
  } else {
    if (speed_source_type_ != "vesc") {
      RCLCPP_WARN(this->get_logger(),
                  "Unknown speed_source_type '%s', falling back to 'vesc'",
                  speed_source_type_.c_str());
    }
    subscription_vesc_ =
        this->create_subscription<vesc_msgs::msg::VescStateStamped>(
            vesc_state_topic_, 10,
            std::bind(&SpeedCompensator::vesc_state_callback, this, _1));
  }

  param_callback_handle_ = this->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &parameter : parameters) {
          apply_runtime_parameter(parameter);
        }
        return result;
      });

  RCLCPP_INFO(this->get_logger(),
              "Speed compensator launched: %s -> %s (speed source: %s, "
              "activation: %.1fm/s, decel gain: %.2f, accel gain: %.2f)",
              input_drive_topic_.c_str(), output_drive_topic_.c_str(),
              speed_source_type_.c_str(), activation_speed_, decel_boost_gain_,
              accel_boost_gain_);
}

// 런타임에 변경 가능한 파라미터 하나를 대응하는 멤버 변수에 반영합니다.
// 처리한 파라미터면 true, 목록에 없는 이름이면 false를 반환합니다.
bool SpeedCompensator::apply_runtime_parameter(
    const rclcpp::Parameter &parameter) {
  const std::string &name = parameter.get_name();

  if (name == "speed_to_erpm_gain") {
    speed_to_erpm_gain_ = parameter.as_double();
  } else if (name == "speed_to_erpm_offset") {
    speed_to_erpm_offset_ = parameter.as_double();
  } else if (name == "activation_speed") {
    activation_speed_ = std::max(0.0, parameter.as_double());
  } else if (name == "activation_ramp_range") {
    activation_ramp_range_ = std::max(0.0, parameter.as_double());
  } else if (name == "decel_boost_gain") {
    decel_boost_gain_ = std::max(0.0, parameter.as_double());
  } else if (name == "accel_boost_gain") {
    accel_boost_gain_ = std::max(0.0, parameter.as_double());
  } else if (name == "decel_boost_max") {
    decel_boost_max_ = std::max(0.0, parameter.as_double());
  } else if (name == "accel_boost_max") {
    accel_boost_max_ = std::max(0.0, parameter.as_double());
  } else if (name == "speed_error_deadband") {
    speed_error_deadband_ = std::max(0.0, parameter.as_double());
  } else if (name == "speed_timeout_sec") {
    speed_timeout_sec_ = std::max(0.0, parameter.as_double());
  } else {
    return false;
  }

  return true;
}

// 목표 속도(target)에 실측 속도 기반 보정을 적용합니다.
double SpeedCompensator::compensate_speed(double target_speed) const {
  // 실측 속도가 없거나 오래됐으면 보정하지 않고 통과 (fail-safe)
  if (!speed_received_ ||
      (this->now() - last_speed_time_).seconds() > speed_timeout_sec_) {
    return target_speed;
  }

  // 후진(음수 목표)에는 관여하지 않음
  if (target_speed < 0.0) {
    return target_speed;
  }

  // activation_speed 미만에서는 동작하지 않음.
  // 경계에서 출력이 점프하지 않도록 activation_speed부터
  // activation_ramp_range 구간에 걸쳐 보정 비율을 0->1로 올림
  const double speed_over_activation =
      std::abs(current_speed_) - activation_speed_;
  if (speed_over_activation <= 0.0) {
    return target_speed;
  }
  const double activation_ratio =
      activation_ramp_range_ > 1e-6
          ? std::min(speed_over_activation / activation_ramp_range_, 1.0)
          : 1.0;

  // 데드밴드 경계에서도 연속이 되도록 유효 오차 = |오차| - 데드밴드 사용
  const double speed_error = current_speed_ - target_speed;
  const double effective_error =
      std::abs(speed_error) - speed_error_deadband_;
  if (effective_error <= 0.0) {
    return target_speed;
  }

  if (speed_error > 0.0) {
    // 감속 중: 오차에 비례해 더 감속
    const double boost = std::min(
        decel_boost_gain_ * effective_error * activation_ratio,
        decel_boost_max_);
    return std::max(0.0, target_speed - boost);
  }

  // 가속 중: 오차에 비례해 더 가속
  const double boost = std::min(
      accel_boost_gain_ * effective_error * activation_ratio,
      accel_boost_max_);
  return target_speed + boost;
}

void SpeedCompensator::drive_callback(
    const ackermann_msgs::msg::AckermannDriveStamped::ConstSharedPtr msg) {
  auto out_msg = *msg;
  out_msg.drive.speed = compensate_speed(msg->drive.speed);

  RCLCPP_DEBUG_THROTTLE(
      this->get_logger(), *this->get_clock(), 200,
      "current: %.2fm/s ... target: %.2fm/s ... output: %.2fm/s",
      current_speed_, msg->drive.speed, out_msg.drive.speed);

  publisher_drive_->publish(out_msg);
}

void SpeedCompensator::vesc_state_callback(
    const vesc_msgs::msg::VescStateStamped::ConstSharedPtr msg) {
  if (std::abs(speed_to_erpm_gain_) < 1e-6) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "speed_to_erpm_gain is zero, ignoring VESC state");
    return;
  }

  // erpm -> m/s 변환 (vesc_to_odom과 동일한 식)
  current_speed_ =
      (msg->state.speed - speed_to_erpm_offset_) / speed_to_erpm_gain_;
  speed_received_ = true;
  last_speed_time_ = this->now();
}

void SpeedCompensator::odom_callback(
    const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
  current_speed_ = msg->twist.twist.linear.x;
  speed_received_ = true;
  last_speed_time_ = this->now();
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node_ptr = std::make_shared<SpeedCompensator>();
  rclcpp::spin(node_ptr);
  rclcpp::shutdown();
  return 0;
}
