#ifndef WAYPOINT_VISUALIZER_HPP
#define WAYPOINT_VISUALIZER_HPP

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
using std::placeholders::_1;
using namespace std::chrono_literals;

class WaypointVisualizer : public rclcpp::Node {
   public:
    WaypointVisualizer();

   private:
    struct PathPoints {
        std::vector<double> X;
        std::vector<double> Y;
    };

    // topic names
    std::string path_topic;
    std::string rviz_waypoints_topic;
    std::string path_frame_id;

    // path data
    PathPoints waypoints;
    bool path_received;

    // Subscriber initialisation
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subscription_path;

    // Publisher initialisation
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_path_pub;

    // Timer initialisation
    rclcpp::TimerBase::SharedPtr timer_;

    // private functions

    void path_callback(const nav_msgs::msg::Path::SharedPtr path_msg);
    void visualize_points();
    void timer_callback();
};

#endif // WAYPOINT_VISUALIZER_HPP
