#include "waypoint_visualizer.hpp"

WaypointVisualizer::WaypointVisualizer()
    : Node("waypoint_visualizer_node"), path_received(false) {
    this->declare_parameter("path_topic", "/Path");
    this->declare_parameter("rviz_waypoints_topic", "/waypoints");

    path_topic = this->get_parameter("path_topic").as_string();
    rviz_waypoints_topic = this->get_parameter("rviz_waypoints_topic").as_string();

    rclcpp::QoS pathQos(rclcpp::KeepLast(10));
    pathQos.reliability(rclcpp::ReliabilityPolicy::Reliable);

    subscription_path = this->create_subscription<nav_msgs::msg::Path>(
        path_topic, pathQos,
        std::bind(&WaypointVisualizer::path_callback, this, std::placeholders::_1));

    vis_path_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(rviz_waypoints_topic, 1000);
    timer_ = this->create_wall_timer(2000ms, std::bind(&WaypointVisualizer::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "Waypoint visualizer node has been launched");
}

void WaypointVisualizer::path_callback(const nav_msgs::msg::Path::SharedPtr path_msg) {
    if (!path_msg || path_msg->poses.empty()) {
        RCLCPP_WARN(this->get_logger(), "Received empty path message, ignoring it.");
        return;
    }

    waypoints.X.clear();
    waypoints.Y.clear();
    path_frame_id = path_msg->header.frame_id.empty() ? "map" : path_msg->header.frame_id;

    for (const auto &pose_stamped : path_msg->poses) {
        waypoints.X.push_back(pose_stamped.pose.position.x);
        waypoints.Y.push_back(pose_stamped.pose.position.y);
    }

    path_received = true;
    visualize_points();
}

void WaypointVisualizer::visualize_points() {
    if (!path_received) {
        return;
    }

    auto marker_array = visualization_msgs::msg::MarkerArray();
    auto marker = visualization_msgs::msg::Marker();
    marker.header.frame_id = path_frame_id.empty() ? "map" : path_frame_id;
    marker.header.stamp = this->now();
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.15;
    marker.scale.y = 0.15;
    marker.scale.z = 0.15;
    marker.color.a = 1.0;
    marker.color.g = 1.0;

    for (unsigned int i = 0; i < waypoints.X.size(); ++i) {
        marker.pose.position.x = waypoints.X[i];
        marker.pose.position.y = waypoints.Y[i];
        marker.pose.position.z = 0.0;
        marker.id = i;
        marker_array.markers.push_back(marker);
    }

    vis_path_pub->publish(marker_array);
}

void WaypointVisualizer::timer_callback() {
    visualize_points();
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node_ptr = std::make_shared<WaypointVisualizer>();  // initialise node pointer
    rclcpp::spin(node_ptr);
    rclcpp::shutdown();
    return 0;
}
