#ifndef PATH_SPEED_FILTER_HPP
#define PATH_SPEED_FILTER_HPP

#include <string>
#include <vector>

#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"

// path 속도 프로파일 후처리 노드 (오픈루프).
// path의 z(속도) 프로파일에서 각 waypoint 앞쪽의 감가속도를 계산해,
// 감속 구간에서는 감속량에 비례해 타겟 속도를 미리 떨어뜨린 path를
// 재발행합니다. activation_speed(m/s) 이상에서만 동작합니다.
//
// 모든 경계(감속 시작/끝, activation 속도, 감속량 상한)는 연속 함수
// (max/min/clamp/선형보간)로만 구성되어 출력에 절단 지점이 없습니다.
class PathSpeedFilter : public rclcpp::Node {
public:
  PathSpeedFilter();

private:
  // 초기화 전용 파라미터
  std::string input_path_topic_;
  std::string output_path_topic_;

  // 런타임 변경 가능 파라미터
  bool path_is_circular_ = true;
  double decel_preview_distance_ = 1.5; // 감가속도를 계산할 전방 거리(m)
  double decel_drop_gain_ = 0.5;  // 감속도 1m/s^2당 떨어뜨리는 속도(m/s)
  double activation_speed_ = 4.0; // 이 프로파일 속도(m/s) 이상에서만 동작
  double activation_ramp_range_ = 1.0; // 보정 비율을 0->1로 올리는 속도 구간(m/s)
  double max_speed_drop_ = 3.0;   // 한 waypoint에서 떨어뜨리는 속도 상한(m/s)

  nav_msgs::msg::Path::ConstSharedPtr last_path_;

  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subscription_path_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_path_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      param_callback_handle_;

  // 런타임 변경 가능 파라미터 1개를 대응 멤버 변수에 반영합니다.
  bool apply_runtime_parameter(const rclcpp::Parameter &parameter);

  void path_callback(const nav_msgs::msg::Path::ConstSharedPtr path_msg);
  void publish_filtered_path();

  // start_idx에서 전방 preview 거리만큼 프로파일을 따라간 지점의 속도를
  // 선형보간으로 구합니다. 실제로 진행한 거리는 traveled에 반환합니다.
  // 프로파일에 유효하지 않은 속도가 있으면 false를 반환합니다.
  bool speed_at_distance_ahead(const std::vector<double> &speeds,
                               const std::vector<double> &segment_lengths,
                               int start_idx, double preview_distance,
                               double *speed_ahead, double *traveled) const;
};

#endif // PATH_SPEED_FILTER_HPP
