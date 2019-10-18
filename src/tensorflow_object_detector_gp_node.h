#pragma once

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <rwth_perception_people_msgs/GroundPlane.h>
#include <spencer_tracking_msgs/DetectedPersons.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "tensorflow_object_detector.h"

class TensorFlowObjectDetectorGPCore
{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    // image_transport::Subscriber imageSubscriber_;
    image_transport::Publisher imagePublisher_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> imageSubscriber_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>> cameraInfoSubscriber_;
    std::shared_ptr<message_filters::Subscriber<rwth_perception_people_msgs::GroundPlane>> groundPlaneSubscriber;
    ros::Publisher pubDetectedPersons;

    using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, rwth_perception_people_msgs::GroundPlane>;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync;

    std::unique_ptr<TensorFlowObjectDetector> detector_;

    //parameters
    double score_threshold_;
    bool always_output_image_;
    std::string camera_ns;
    std::string ground_plane;
    std::string pub_topic_detected_persons;
    int queue_size;
    double world_scale, pose_variance;
    int detection_id_increment, detection_id_offset, current_detection_id;

    void getRay(const Eigen::Matrix3d& K, const Eigen::Vector3d& x, Eigen::Vector3d& ray1, Eigen::Vector3d& ray2);
    void intersectPlane(const Eigen::Vector3d& gp, double gpd, const Eigen::Vector3d& ray1, const Eigen::Vector3d& ray2, Eigen::Vector3d& point);
    void calc3DPosFromBBox(const Eigen::Matrix3d& K, const Eigen::Vector3d& GPN_, double GPD_, double x, double y, double w, double h, double ConvertScale, Eigen::Vector3d& pos3D);

public:
    TensorFlowObjectDetectorGPCore(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
    ~TensorFlowObjectDetectorGPCore() = default;
    
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg, const sensor_msgs::CameraInfoConstPtr &camera_info, const rwth_perception_people_msgs::GroundPlaneConstPtr &g);

    void run()
    {
        ros::spin();
    }

    TensorFlowObjectDetectorGPCore(const TensorFlowObjectDetectorGPCore&) = delete;
    TensorFlowObjectDetectorGPCore(TensorFlowObjectDetectorGPCore&&) = delete;
    TensorFlowObjectDetectorGPCore& operator=(const TensorFlowObjectDetectorGPCore&) = delete;
    TensorFlowObjectDetectorGPCore& operator=(TensorFlowObjectDetectorGPCore&&) = delete;
};
