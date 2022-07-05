#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include "AHCPlaneFitter.hpp"

struct OrganizedImage3D {
    OrganizedImage3D(const cv::Mat& c) : cloud(c) {}
    inline int width() const { return cloud.cols; }
    inline int height() const { return cloud.rows; }

    bool get(const int row, const int col, double& x, double& y, double& z) const {
        const cv::Vec3f& point = cloud.at<cv::Vec3f>(row, col);
        x = point[0], y = point[1], z = point[2];
        if (std::isnan(x) || std::isnan(y) || std::isnan(z) ||
           (x == 0 && y == 0 && z == 0)) {
            return false;
        }
        return true;
    }
    const double unitScaleFactor = 1;
    const cv::Mat& cloud;
};

typedef ahc::PlaneFitter<OrganizedImage3D> PlaneFitter;

namespace segmenter {
struct CameraIntrinsics {
    float fx, fy, cx, cy;
    uint32_t width, height;
    cv::Mat K;

    CameraIntrinsics() {}
    CameraIntrinsics(cv::Mat &cam_intrinsics, uint32_t w, uint32_t h) :
        K(cam_intrinsics), width(w), height(h) {}
    CameraIntrinsics(float fx, float fy, float cx, float cy, uint32_t w, uint32_t h) :
        fx(fx), fy(fy), cx(cx), cy(cy), width(w), height(h) {
        K = cv::Mat::eye(cv::Size(3, 3), CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
    }
};

struct Segment {
    cv::Mat semantic_mask;
    uint8_t semantic_labels;
    std::vector<cv::Vec3f> points;
};

struct SegmenterConfig {
    double min_area;
    double open_kernel_size;
    double distance_thres;
    bool debug_show;
};

struct PlaneFitterConfig {
    int minSupport;
    int windowWidth;
    int windowHeight;
    bool doRefine;

    int initType;

    //T_mse
    double stdTol_merge;
    double stdTol_init;
    double depthSigma;

    //T_dz
    double depthAlpha;
    double depthChangeTol;

    //T_ang
    double z_near;
    double z_far;
    double angle_near;
    double angle_far;
    double similarityTh_merge;
    double similarityTh_refine;
};

class DepthSegmenter {
public:
    DepthSegmenter() {}
    DepthSegmenter(cv::Mat &K, uint32_t w, uint32_t h, struct SegmenterConfig &segmenter_config,
        struct PlaneFitterConfig & plane_config) {
        initAHCParams(plane_config);
        camera_intrinsics_ = CameraIntrinsics(K, w, h);
        min_area_ = segmenter_config.min_area;
        open_kernel_size_ = segmenter_config.open_kernel_size;
        distance_thres_ = segmenter_config.distance_thres;
        debug_show_ = segmenter_config.debug_show;
    }

    DepthSegmenter(float fx, float fy, float cx, float cy, uint32_t w, uint32_t h) {
        initAHCParams();
        camera_intrinsics_ = CameraIntrinsics(fx, fy, cx, cy, w, h);
        min_area_ = 1000;
        open_kernel_size_ = 5;
        distance_thres_ = 0.08;
        debug_show_ = true;
    }

    void processFrame(const cv::Mat& depth_image) {
        cv::Mat depth_map = computeDepthMap(depth_image);
        cv::Mat plane_seg = cv::Mat(depth_image.size(), CV_8UC3, 0.0f);
        std::vector< std::vector<int> > membership;
        OrganizedImage3D xyz(depth_map);
        pf_.run(&xyz, &membership, &plane_seg);

        // threshold depth image
        cv::Mat rescaled_depth;
        cv::Mat threshold_image(depth_image.size(), CV_32FC1);
        depth_image.convertTo(rescaled_depth, CV_32FC1);    
        constexpr float kMaxBinaryValue = 1.0f;
        constexpr double kNanThreshold = 0.0;
        cv::threshold(rescaled_depth, threshold_image, kNanThreshold, kMaxBinaryValue, cv::THRESH_BINARY);

        // get un-plane threshold image
        cv::Mat rescaled_plane_seg = cv::Mat(threshold_image.size(), CV_32FC1, cv::Scalar(0));
        cv::Mat un_plane_thres(threshold_image.size(), CV_32FC1);
        cv::cvtColor(plane_seg, plane_seg, cv::COLOR_RGB2GRAY);
        plane_seg.convertTo(plane_seg, CV_32FC1);
        cv::threshold(plane_seg, rescaled_plane_seg, kNanThreshold, kMaxBinaryValue, cv::THRESH_BINARY);
        std::cout << "threshold_image.channels() = " << threshold_image.channels() << std::endl;
        std::cout << "rescaled_plane_seg.channels() = " << rescaled_plane_seg.channels() << std::endl;
        un_plane_thres = threshold_image - rescaled_plane_seg;

        // open operator
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(open_kernel_size_, open_kernel_size_));
        cv::morphologyEx(un_plane_thres, un_plane_thres, cv::MORPH_OPEN, element);

        // delete some bad point
        for (size_t i = 0; i < un_plane_thres.rows; ++i) {
            for (size_t j = 0; j < un_plane_thres.cols; ++j) {
                if (un_plane_thres.at<float>(i, j) == 0) {
                    continue;
                }
                cv::Vec3f point = depth_map.at<cv::Vec3f>(i, j);
                if (checkPointDistanceToPlane(point, distance_thres_)) {
                    un_plane_thres.at<float>(i, j) = 0;
                }
            }
        }

        // label map
        std::vector<Segment> segments;
        std::vector<cv::Mat> segment_masks;
        labelMap(depth_map, un_plane_thres, membership, segment_masks, segments);
    }
    
private:
    cv::Mat computeDepthMap(const cv::Mat& depth_image) {
        cv::Mat rescaled_depth = cv::Mat::zeros(depth_image.size(), CV_32FC1);
        if (depth_image.type() == CV_16UC1) {
            std::cout << "depth_image.type() == CV_16UC1" << std::endl;
            cv::rgbd::rescaleDepth(depth_image, CV_32FC1, rescaled_depth);
        } else if (depth_image.type() == CV_32FC1) {
            std::cout << "depth_image.type() == CV_32FC1" << std::endl;
            depth_image.copyTo(rescaled_depth);
        } else {
            throw "unknown depth image type.";
        }

        constexpr double kZeroValue = 0.0;
        cv::Mat nan_mask = rescaled_depth != rescaled_depth;
        rescaled_depth.setTo(kZeroValue, nan_mask);
        
        cv::Mat depth_map = cv::Mat::zeros(depth_image.size(), CV_32FC3);
        cv::rgbd::depthTo3d(rescaled_depth, camera_intrinsics_.K, depth_map);

        return depth_map;
    }

    void generateRandomColorsAndLabels(size_t contours_size, std::vector<cv::Scalar> &colors, std::vector<int> &labels) {
        for (size_t i = colors.size(); i < contours_size; ++i) {
            labels.push_back(i);
            colors.push_back(
                cv::Scalar(255 * (rand() / static_cast<float>(RAND_MAX)),
                           255 * (rand() / static_cast<float>(RAND_MAX)),
                           255 * (rand() / static_cast<float>(RAND_MAX))));
        }
    }

    void labelMap(cv::Mat &depth_map, cv::Mat &un_plane, std::vector<std::vector<int>> &membership,
        std::vector<cv::Mat> &segment_masks, std::vector<Segment> &segments) {
        cv::Mat output = cv::Mat::zeros(depth_map.size(), CV_8UC3);
        constexpr size_t kMaskValue = 255u;
        
        // process un_plane segment
        static const cv::Point kContourOffset = cv::Point(0, 0);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::Mat un_plane_8u;
        un_plane.convertTo(un_plane_8u, CV_8U);
        cv::findContours(un_plane_8u, contours, hierarchy, cv::RETR_TREE, CV_CHAIN_APPROX_NONE, kContourOffset);

        // get colors and labels
        int plane_color_start_index = contours.size();
        std::vector<cv::Scalar> colors;
        std::vector<int> labels;
        generateRandomColorsAndLabels(contours.size() + membership.size(), colors, labels);

        // process small area
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            constexpr int kNoParentContour = -1;
            if (area < min_area_) {
                const int parent_contour = hierarchy[i][3];
                if (parent_contour == kNoParentContour) {
                    colors[i] = cv::Scalar(0, 0, 0);
                    labels[i] = -1;
                    cv::drawContours(un_plane_8u, contours, i, cv::Scalar(0), CV_FILLED, 8, hierarchy);
                } else {
                    if (hierarchy[i][0] == -1 && hierarchy[i][1] == -1) {
                        // Assign the color of the parent contour.
                        colors[i] = colors[parent_contour];
                        labels[i] = labels[parent_contour];
                    } else {
                        colors[i] = cv::Scalar(0, 0, 0);
                        labels[i] = -1;
                        drawContours(un_plane_8u, contours, i, cv::Scalar(0u), CV_FILLED, 8, hierarchy);
                    }
                }
            }
        }

        cv::Mat output_labels = cv::Mat(depth_map.size(), CV_32SC1, cv::Scalar(0));
        for (size_t i = 0; i < contours.size(); ++i) {
            cv::drawContours(output, contours, i, cv::Scalar(colors[i]), CV_FILLED, 8, hierarchy);
            cv::drawContours(output_labels, contours, i, cv::Scalar(labels[i]), CV_FILLED, 8, hierarchy);
            cv::drawContours(un_plane_8u, contours, i, cv::Scalar(0u), 1, 8, hierarchy);
        }

        output.setTo(cv::Scalar(0, 0, 0), un_plane_8u == 0);
        output_labels.setTo(-1, un_plane_8u == 0);

        // Create a map of all the labels.
        size_t value = 0u;
        std::map<size_t, size_t> labels_map;
        for (size_t i = 0u; i < labels.size(); ++i) {
            if (labels[i] >= 0) {
                // Create a new map if label is not yet in keys.
                if (labels_map.find(labels[i]) == labels_map.end()) {
                    labels_map[labels[i]] = value;
                    ++value;
                }
            }
        }

        // compute non-plane segments
        segments.resize(labels_map.size());
        segment_masks.resize(labels_map.size());
        for (cv::Mat& segment_mask : segment_masks) {
            segment_mask = cv::Mat(depth_map.size(), CV_8UC1, cv::Scalar(0));
        }

        for (size_t i = 0; i < output_labels.rows; ++i) {
            for (size_t j = 0; j < output_labels.cols; ++j) {
                int label = output_labels.at<int>(i, j);
                if (label < 0) {
                    continue;
                } else {
                    cv::Vec3f point = depth_map.at<cv::Vec3f>(i, j);
                    Segment &segment = segments[labels_map.at(label)];
                    cv::Mat &semantic_mask = segment_masks[labels_map.at(label)];
                    segment.points.push_back(point);
                    semantic_mask.at<uint8_t>(i, j) = kMaskValue;
                    output.at<cv::Vec3b>(i, j) = cv::Vec3b(colors[label][0],
                                                           colors[label][1],
                                                           colors[label][2]);
                }
            }
        }

        // process plane segment
        for (size_t i = 0; i < membership.size(); ++i) {
            ahc::PlaneSeg::shared_ptr plane = pf_.extractedPlanes[i];
            std::vector<int> &planeIndexs = membership[i];
            for (size_t j = 0; j < planeIndexs.size(); ++j) {
                int pixies = planeIndexs[j];
                int pa = pixies / pf_.width;
                int pb = pixies - pa * pf_.width;
                output.at<cv::Vec3b>(pa, pb) = cv::Vec3b(colors[i + plane_color_start_index][0],
                                                         colors[i + plane_color_start_index][1],
                                                         colors[i + plane_color_start_index][2]);
            }
        }

        if (debug_show_) {
            cv::imshow("label_map", output);
            cv::waitKey(1);
        }
    }

    void initAHCParams(const struct PlaneFitterConfig &config) {
        pf_.minSupport = config.minSupport;
        pf_.windowWidth = config.windowWidth;
        pf_.windowHeight = config.windowHeight;
        pf_.doRefine = config.doRefine;

        pf_.params.initType = (ahc::InitType)config.initType;

        //T_mse
        pf_.params.stdTol_merge = config.stdTol_merge;
        pf_.params.stdTol_init = config.stdTol_init;
        pf_.params.depthSigma = config.depthSigma;

        //T_dz
        pf_.params.depthAlpha = config.depthAlpha;
        pf_.params.depthChangeTol = config.depthChangeTol;

        //T_ang
        pf_.params.z_near = config.z_near;
        pf_.params.z_far = config.z_far;
        pf_.params.angle_near = MACRO_DEG2RAD(config.angle_near);
        pf_.params.angle_far = MACRO_DEG2RAD(config.angle_far);
        pf_.params.similarityTh_merge = std::cos(MACRO_DEG2RAD(config.similarityTh_merge));
        pf_.params.similarityTh_refine = std::cos(MACRO_DEG2RAD(config.similarityTh_refine));
    }

    void initAHCParams() {
        pf_.minSupport = 1000;
        pf_.windowWidth = 30;
        pf_.windowHeight = 30;
        pf_.doRefine = 1;

        pf_.params.initType = ahc::INIT_STRICT;

        //T_mse
        pf_.params.stdTol_merge = 0.008;
        pf_.params.stdTol_init = 0.005;
        pf_.params.depthSigma = 1.6e-3;

        //T_dz
        pf_.params.depthAlpha = 0.04;
        pf_.params.depthChangeTol = 0.02;

        //T_ang
        pf_.params.z_near = 0.2;
        pf_.params.z_far = 3.0;
        pf_.params.angle_near = MACRO_DEG2RAD(45);
        pf_.params.angle_far = MACRO_DEG2RAD(90);
        pf_.params.similarityTh_merge = std::cos(MACRO_DEG2RAD(60));
        pf_.params.similarityTh_refine = std::cos(MACRO_DEG2RAD(45));
    }

    bool checkPointDistanceToPlane(const cv::Vec3f &point, double thres) {
        for (size_t i = 0; i < pf_.extractedPlanes.size(); ++i) {
            double *center = pf_.extractedPlanes[i]->center;
            double *normal = pf_.extractedPlanes[i]->normal;
            double a = std::abs(normal[0] * (center[0] - point[0]) +
                                normal[1] * (center[1] - point[1]) +
                                normal[2] * (center[2] - point[2]));
            double b = std::sqrt(normal[0] * normal[0] +
                                 normal[1] * normal[1] +
                                 normal[2] * normal[2]);
            double distance = a / b;
            if (distance < thres) {
                return true;
            }
        }
        return false;
    }
    
    PlaneFitter pf_;

    // config parameters
    struct CameraIntrinsics camera_intrinsics_;
    double min_area_;
    double open_kernel_size_;
    double distance_thres_;
    bool debug_show_;
};
} // segmenter