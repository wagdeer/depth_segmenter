#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <opencv2/opencv.hpp>
#include "AHCPlaneFitter.hpp"
#include "DepthSegmenter.hpp"

class DepthSegmenterNode {
public:
    DepthSegmenterNode() :
        node_handle_("~") {
        node_handle_.param<std::string>("depth_image_sub_topic", depth_image_topic_, depth_image_topic_);
        depth_segmenter_ = initSegmenter();
        sub_img_ = node_handle_.subscribe<sensor_msgs::Image>(depth_image_topic_, 1, &DepthSegmenterNode::imgCallback, this);
    }

private:
    segmenter::DepthSegmenter initSegmenter() {
        float fx, fy, cx, cy, width, heigth;

        node_handle_.param<float>("fx", fx, fx);
        node_handle_.param<float>("fy", fy, fy);
        node_handle_.param<float>("cx", cx, cx);
        node_handle_.param<float>("cy", cy, cy);
        node_handle_.param<float>("width", width, width);
        node_handle_.param<float>("heigth", heigth, heigth);

        segmenter::DepthSegmenter depth_segmenter(fx, fy, cx, cy, width, heigth);
        return depth_segmenter;
    }

    void imgCallback(const sensor_msgs::Image::ConstPtr& depth_msg) {
        cv_bridge::CvImagePtr cv_depth_image(new cv_bridge::CvImage);
        if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
            cv_depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
            cv_depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        cv::Mat depth_image = cv_depth_image->image;

        depth_segmenter_.processFrame(depth_image);
    }

    segmenter::DepthSegmenter depth_segmenter_;
    ros::NodeHandle node_handle_;
    ros::Subscriber sub_img_;
    std::string depth_image_topic_;
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "depth_segmenter");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    
    DepthSegmenterNode node;

    while (ros::ok()) {
        ros::spin();
    }

    return 0;
}