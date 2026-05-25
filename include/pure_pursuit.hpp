#ifndef PURE_PURSUIT_HPP
#define PURE_PURSUIT_HPP

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
#include "std_msgs/msg/u_int16_multi_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#define _USE_MATH_DEFINES
using std::placeholders::_1;
using namespace std::chrono_literals;

class PurePursuit : public rclcpp::Node {
public:
  PurePursuit();

private:
  // 구조체: 추종할 경로의 데이터와 관련된 상태를 보관합니다. ROS의 nav_msgs::Path
  // 메시지로부터 읽어온 좌표를 저장하고, 현재 추종중인 포인트의 인덱스를
  // 관리합니다. z 값은 속도(v)로 사용합니다.
  struct Waypoints {
    // 각 축별 좌표와 속도 벡터
    std::vector<double> X;
    std::vector<double> Y;
    std::vector<double> V;
    // 현재 lookahead waypoint 인덱스
    int index = 0;
    // 현재 차량 위치에 가장 가까운 waypoint 인덱스
    int velocity_index = -1;
    // 속도 프로파일을 읽을 waypoint 인덱스
    int speed_index = -1;
  };

  // Waypoints 구조체 인스턴스
  Waypoints waypoints;

  // steering waypoint(lookahead)와 속도 포인트 인덱스는 Waypoints 내부에서
  // 관리합니다. 기존 멤버는 남겨두되 사용하지 않습니다.
  int lookahead_index;
  int velocity_index;

  // lookahead 및 현재 포인트의 좌표 (world, car 프레임)
  Eigen::Vector3d lookahead_point_world; // 월드 좌표계 (보통 "map")
  Eigen::Vector3d lookahead_point_car;   // 차량 좌표계
  Eigen::Vector3d current_point_world;   // 차량에 가장 가까운 waypoint
  Eigen::Vector3d speed_point_world;     // 속도 프로파일을 읽는 waypoint

  //차량 현재 위치(map프레임)
  double x_car_world;
  double y_car_world;
  //장애물 위치(map프레임)
  double x_obs = 0;
  double y_obs = 0;

  std::string odom_topic;
  std::string car_refFrame;
  std::string drive_topic;
  std::string drive_test_topic;
  std::string drive_output_topic;
  std::string global_refFrame;
  std::string path_topic;
  std::string rviz_current_waypoint_topic;
  std::string rviz_lookahead_waypoint_topic;
  std::string rviz_speed_offset_waypoint_topic;
  std::string rviz_runtime_params_topic;
  std::string rf_topic;
  double rviz_runtime_params_x = 0.0;
  double rviz_runtime_params_y = 0.0;
  double rviz_runtime_params_z = 1.2;
  double K_p;
  double K_d;
  double K_i; // PD 제어를 위한 미분(derivative) 게인
  double min_lookahead;
  double max_lookahead;
  double lookahead_ratio;
  double speed_profile_distance_offset;
  double steering_limit;
  double velocity_percentage;
  double max_speed_limit_percentage = 1.0;
  double observed_path_max_speed = 0.0;
  double velocity_reduce_obs = 1;
  double heading_error_gain;
  double steer_reduction_speed_threshold;
  double steer_reduction_constant_coef;
  double steer_reduction_linear_coef;
  double steer_reduction_min_scale;
  double speed_reduction_angle_threshold;
  double max_allowed_steer_drop;
  double speed_reduction_adjust;
  double speed_reduction_prev_scale;
  double previous_speed_reduction;
  double curr_velocity = 0.0;
  bool test_mode = false;
  int rf_speed_scale_channel = 6;
  int rf_max_limit_channel = 7;
  int rf_enable_channel = 8;
  int rf_enable_threshold = 750;
  int rf_value_min = 0;
  int rf_value_max = 1500;
  bool rf_runtime_control_active = false;
  double drive_output_rate_hz = 50.0;
  double active_drive_output_rate_hz = 0.0;
  double steer_latest_blend = 0.10;
  double steer_large_change_blend = 0.55;
  double steer_blend_change_threshold_deg = 10.0;
  double steer_speed_filter_start_speed = 0.0;
  double steer_speed_filter_end_speed = 0.0;
  double steer_speed_filter_final_blend = 0.10;
  double speed_latest_blend = 0.90;
  double target_steer = 0.0;
  double target_speed = 0.0;
  double output_steer = 0.0;
  double output_speed = 0.0;
  double current_lookahead_distance = 0.0;
  bool has_target_command_ = false;
  bool output_command_initialized_ = false;
  int min_searching_idx_offset;
  int max_searching_idx_offset;
  double car_heading;
  bool path_is_circular = true;
  double steering_expo_gain;
  double steering_expo_curve;
  bool slow_with_obs;
  double slow_th_dist;
  double slow_amount;

  bool emergency_breaking = false;
  std::string lane_number = "left"; // "left" 또는 "right"

  // Waypoint 개수 (path 길이). Waypoints 구조체의 벡터 크기와 동일합니다.
  int num_waypoints;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr drive_output_timer_;
  rclcpp::TimerBase::SharedPtr runtime_param_visualization_timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      runtime_param_callback_handle_;

  // Subscriber
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_odom;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr subscription_odom_obs;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr obs_status;

  // Path subscriber: nav_msgs::Path 타입의 경로 메시지를 수신합니다.
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subscription_path;
  rclcpp::Subscription<std_msgs::msg::UInt16MultiArray>::SharedPtr
      subscription_rf;

  // Publisher
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      publisher_drive;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      vis_current_point_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      vis_lookahead_point_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      vis_speed_point_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      vis_runtime_params_pub;

  // PD 제어를 위한 이전 오차와 이전 시간 (미분 항 계산용)
  double prev_error;
  rclcpp::Time prev_time;
  double integral_error;

  // 경로 수신 여부 플래그. 초기에는 false이며, path_callback 에서 유효한
  // 경로를 수신하면 true 로 설정됩니다. 경로가 설정되기 전에는
  // 제어 로직을 수행하지 않습니다.
  bool path_received_;
  // private 함수들
  double to_radians(double degrees);
  double to_degrees(double radians);
  double p2pdist(const double &x1, const double &x2, const double &y1,
                 const double &y2);

  double apply_steering_expo(double steering_angle, double steering_limit_rad);
  double current_max_speed_limit() const;
  double apply_max_speed_limit(double speed) const;
  double rf_raw_to_unit(int raw_value) const;
  bool read_rf_channel(const std_msgs::msg::UInt16MultiArray &rf_msg,
                       int channel_index, int *raw_value) const;
  void set_runtime_percentages(double velocity_scale,
                               double max_speed_limit_scale);
  Eigen::Vector3d sample_path_point_by_distance(int start_idx, double distance,
                                                int *reached_idx = nullptr);
  std::string selected_drive_topic() const;
  void configure_drive_publisher();
  void configure_drive_output_timer();

  void visualize_lookahead_point(Eigen::Vector3d &point);
  void visualize_current_point(Eigen::Vector3d &point);
  void visualize_speed_point(Eigen::Vector3d &point);
  void visualize_runtime_params();

  void get_waypoint();

  void transformandinterp_waypoint();

  double p_controller();

  double get_velocity(double steering_angle);

  void update_target_command(double steering_angle);
  void drive_output_timer_callback();

  void
  odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_submsgObj);
  void obs_odom_callback(const geometry_msgs::msg::PointStamped msg);
  void obs_status_callback(const geometry_msgs::msg::PointStamped msg);
  void rf_callback(const std_msgs::msg::UInt16MultiArray::ConstSharedPtr rf_msg);

  void timer_callback();

  // nav_msgs::Path 토픽 수신을 처리하는 콜백. 수신된 경로를 내부 Waypoints
  // 구조체에 저장합니다. position.z는 속도(v)로 활용됩니다.
  void path_callback(const nav_msgs::msg::Path::SharedPtr path_msg);

  void get_waypoint_new();

  int path_idx_limiter(int idx);
  int advance_index_by_distance(int start_idx, double distance);
  double normalize_angle(double angle);
};

#endif // PURE_PURSUIT_HPP
