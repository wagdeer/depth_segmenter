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

        node_handle_.param<double>("min_area", segmenter_config_.min_area, segmenter_config_.min_area);
        node_handle_.param<double>("open_kernel_size", segmenter_config_.open_kernel_size, segmenter_config_.open_kernel_size);
        node_handle_.param<double>("distance_thres", segmenter_config_.distance_thres, segmenter_config_.distance_thres);
        node_handle_.param<bool>("debug_show", segmenter_config_.debug_show, segmenter_config_.debug_show);

        segmenter::DepthSegmenter depth_segmenter(fx, fy, cx, cy, width, heigth);
        return depth_segmenter;
    }

    void initPlaneFitter() {
        node_handle_.param<int>("minSupport", plane_config_.minSupport, plane_config_.minSupport);
        node_handle_.param<int>("windowWidth", plane_config_.windowWidth, plane_config_.windowWidth);
        node_handle_.param<int>("windowHeight", plane_config_.windowHeight, plane_config_.windowHeight);
        node_handle_.param<bool>("doRefine", plane_config_.doRefine, plane_config_.doRefine);

        node_handle_.param<int>("initType", plane_config_.initType, plane_config_.initType);

        //T_mse
        node_handle_.param<double>("stdTol_merge", plane_config_.stdTol_merge, plane_config_.stdTol_merge);
        node_handle_.param<double>("stdTol_init", plane_config_.stdTol_init, plane_config_.stdTol_init);
        node_handle_.param<double>("depthSigma", plane_config_.depthSigma, plane_config_.depthSigma);

        //T_dz
        node_handle_.param<double>("depthAlpha", plane_config_.depthAlpha, plane_config_.depthAlpha);
        node_handle_.param<double>("depthChangeTol", plane_config_.depthChangeTol, plane_config_.depthChangeTol);

        //T_ang
        node_handle_.param<double>("z_near", plane_config_.z_near, plane_config_.z_near);
        node_handle_.param<double>("z_far", plane_config_.z_far, plane_config_.z_far);
        node_handle_.param<double>("angle_near", plane_config_.angle_near, plane_config_.angle_near);
        node_handle_.param<double>("angle_far", plane_config_.angle_far, plane_config_.angle_far);
        node_handle_.param<double>("similarityTh_merge", plane_config_.similarityTh_merge, plane_config_.similarityTh_merge);
        node_handle_.param<double>("similarityTh_refine", plane_config_.similarityTh_refine, plane_config_.similarityTh_refine);
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
    struct segmenter::PlaneFitterConfig plane_config_;
    struct segmenter::SegmenterConfig segmenter_config_;
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