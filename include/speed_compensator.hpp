#ifndef SPEED_COMPENSATOR_HPP
#define SPEED_COMPENSATOR_HPP

#include <string>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vesc_msgs/msg/vesc_state_stamped.hpp"

// drive 명령 후처리 노드.
// VESC 실측 속도와 컨트롤러의 목표 속도를 비교하여,
//  - 감속 중(실측 > 목표)이면 속도 오차에 비례해 목표 속도를 더 낮추고
//  - 가속 중(실측 < 목표)이면 속도 오차에 비례해 목표 속도를 더 높여
// 재발행합니다. activation_speed(m/s) 이상에서만 동작하며,
// 감속/가속 보정 게인은 각각 따로 설정합니다.
class SpeedCompensator : public rclcpp::Node {
public:
  SpeedCompensator();

private:
  // 초기화 전용 파라미터 (토픽/소스 설정)
  std::string input_drive_topic_;
  std::string output_drive_topic_;
  std::string speed_source_type_; // "vesc" 또는 "odom"
  std::string vesc_state_topic_;
  std::string odom_topic_;

  // 런타임 변경 가능 파라미터
  double speed_to_erpm_gain_ = 3172.47;
  double speed_to_erpm_offset_ = 0.0;
  double activation_speed_ = 4.0;    // 이 실측 속도(m/s) 이상에서만 보정 동작
  double decel_boost_gain_ = 0.0;    // 감속 보정 비례 게인 (오차 1m/s당 추가 감속량)
  double accel_boost_gain_ = 0.0;    // 가속 보정 비례 게인 (오차 1m/s당 추가 가속량)
  double decel_boost_max_ = 3.0;     // 최대 추가 감속량(m/s)
  double accel_boost_max_ = 2.0;     // 최대 추가 가속량(m/s)
  double speed_error_deadband_ = 0.1; // 이 이하의 속도 오차(m/s)는 무시
  double speed_timeout_sec_ = 0.5;   // 속도 데이터가 이보다 오래되면 무보정 통과

  // 실측 속도 상태
  double current_speed_ = 0.0;
  bool speed_received_ = false;
  rclcpp::Time last_speed_time_;

  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      subscription_drive_;
  rclcpp::Subscription<vesc_msgs::msg::VescStateStamped>::SharedPtr
      subscription_vesc_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_odom_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      publisher_drive_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      param_callback_handle_;

  bool apply_runtime_parameter(const rclcpp::Parameter &parameter);
  double compensate_speed(double target_speed) const;

  void drive_callback(
      const ackermann_msgs::msg::AckermannDriveStamped::ConstSharedPtr msg);
  void vesc_state_callback(
      const vesc_msgs::msg::VescStateStamped::ConstSharedPtr msg);
  void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
};

#endif // SPEED_COMPENSATOR_HPP
