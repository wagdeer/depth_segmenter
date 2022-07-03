#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <opencv2/opencv.hpp>
#include "AHCPlaneFitter.hpp"
#include "DepthSegmenter.hpp"

#define IMAGE_TOPIC "/camera/depth/image_raw" // TODO: give something
segmenter::DepthSegmenter *g_depth_segmenter;


void img_callback(const sensor_msgs::Image::ConstPtr& depth_msg)
{
    cv_bridge::CvImagePtr cv_depth_image(new cv_bridge::CvImage);
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        cv_depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        cv_depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    cv::Mat depth_image = cv_depth_image->image;
    g_depth_segmenter->processFrame(depth_image);
}

segmenter::DepthSegmenter init_segmenter(ros::NodeHandle &n)
{
    float fx, fy, cx, cy, width, heigth;
    n.param<float>("fx", fx, fx);
    n.param<float>("fy", fy, fy);
    n.param<float>("cx", cx, cx);
    n.param<float>("cy", cy, cy);
    n.param<float>("width", width, width);
    n.param<float>("heigth", heigth, heigth);
    segmenter::DepthSegmenter depth_segmenter(fx, fy, cx, cy, width, heigth);
    
    return depth_segmenter;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "depth_segmenter");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    segmenter::DepthSegmenter depth_segmenter = init_segmenter(n);
    g_depth_segmenter = &depth_segmenter;
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
    
    ros::spin();

    return 0;
}