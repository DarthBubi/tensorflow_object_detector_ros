#include <sstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "tensorflow_object_detector_gp_node.h"

TensorFlowObjectDetectorGPCore::TensorFlowObjectDetectorGPCore(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
  : nh_(nh_private), it_(nh) 
{
    //params
    std::string graph_path, labels_path;
    nh_.param<std::string>("graph_path", graph_path, "");
    nh_.param<std::string>("labels_path", labels_path, "");
    nh_.param<double>("score_threshold", score_threshold_, 0.8);
    nh_.param<bool>("always_output_image", always_output_image_, false);
    nh_.param<std::string>("camera_namespace", camera_ns, std::string("/camera"));
    nh_.param<std::string>("ground_plane", ground_plane, "");
    nh_.param<int>("queue_size", queue_size, 10);
    nh_.param<double>("world_scale", world_scale, 1.0);
    nh_.param<std::string>("detected_persons", pub_topic_detected_persons, std::string("/detected_persons"));
    nh_.param("detection_id_increment", detection_id_increment, 1);
    nh_.param("detection_id_offset",    detection_id_offset, 0);
    nh_.param("pose_variance", pose_variance, 0.05);

    //subscribers
    // imageSubscriber_ = it_.subscribe("image_in", 1, &TensorFlowObjectDetectorGPCore::imageCallback, this);
    imageSubscriber_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh_, camera_ns + "/rgb/image_rect_color", 1));
    cameraInfoSubscriber_.reset(new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh_, camera_ns + "/rgb/camera_info", 1));
    groundPlaneSubscriber.reset(new message_filters::Subscriber<rwth_perception_people_msgs::GroundPlane>(nh_, ground_plane, 1));

    sync.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(queue_size), *imageSubscriber_,
                                                             *cameraInfoSubscriber_, *groundPlaneSubscriber));
    sync->registerCallback(boost::bind(&TensorFlowObjectDetectorGPCore::imageCallback, this, _1, _2, _3));

    //publishers
    imagePublisher_ = it_.advertise("image_out", 1);
    pubDetectedPersons = nh_.advertise<spencer_tracking_msgs::DetectedPersons>(pub_topic_detected_persons, 10);

    //initialize tensorflow
    detector_.reset(new TensorFlowObjectDetector(graph_path, labels_path));
}

void TensorFlowObjectDetectorGPCore::getRay(const Eigen::Matrix3d& K, const Eigen::Vector3d& x, Eigen::Vector3d& ray1, Eigen::Vector3d& ray2)
{
    Eigen::Matrix3d Kinv = K.inverse();

    ray1 = Eigen::Vector3d().Zero();

    Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
    rot *= Kinv;
    ray2 = rot * x;
}

void TensorFlowObjectDetectorGPCore::intersectPlane(const Eigen::Vector3d& gp, double gpd, const Eigen::Vector3d& ray1, const Eigen::Vector3d& ray2, Eigen::Vector3d& point)
{
    Eigen::Vector3d diff = ray1 - ray2;
    double den = gp.dot(diff);
    double t = (gp.dot(ray1) + gpd) / den;
    
    point = ray1;
    diff = (ray2 - ray1) * t;
    point += diff;
}

void TensorFlowObjectDetectorGPCore::calc3DPosFromBBox(const Eigen::Matrix3d& K, const Eigen::Vector3d& GPN_, double GPD_, double x, double y, double w, double h, double ConvertScale, Eigen::Vector3d& pos3D)
{
    // bottom_center is point of the BBOX
    Eigen::Vector3d bottom_center = Eigen::Vector3d().Ones();
    bottom_center(0) = x + w/2.0;
    bottom_center(1) = y + h;

    // Backproject through base point
    Eigen::Vector3d ray_bot_center_1;
    Eigen::Vector3d ray_bot_center_2;
    getRay(K, bottom_center, ray_bot_center_1, ray_bot_center_2);
    
    // Intersect with ground plane
    Eigen::Vector3d gpPointCenter;
    intersectPlane(GPN_, GPD_, ray_bot_center_1, ray_bot_center_2, gpPointCenter);
       
    // Compute 3D Position of BBOx
    pos3D = gpPointCenter * ConvertScale;
}

void TensorFlowObjectDetectorGPCore::imageCallback(const sensor_msgs::Image::ConstPtr& msg, const sensor_msgs::CameraInfoConstPtr &camera_info,
                                                   const rwth_perception_people_msgs::GroundPlaneConstPtr &gp)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8); //for tensorflow, using RGB8
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
        return;
    }

    // perform actual detection
    std::vector<TensorFlowObjectDetector::Result> results;
    try
    {
        results = detector_->detect(cv_ptr->image, score_threshold_);
    }
    catch (std::runtime_error& e)
    {
        ROS_ERROR_STREAM("TensorFlow runtime detection error: " << e.what());
        return;
    }

    // filter by roi and groundplane

    // image output
    cv_bridge::CvImage outImage;;
    outImage.header = cv_ptr->header;
    outImage.encoding = cv_ptr->encoding;
    outImage.image = cv_ptr->image.clone();

    auto height = cv_ptr->image.rows;
    auto width = cv_ptr->image.cols;

    //draw detection rect
    for (const auto& result : results)
    {
        if (result.label_index != 1)
            continue;
        static const cv::Scalar color(255, 0, 0);

        auto topleft = result.box.min();
        auto sizes = result.box.sizes();
        cv::Rect rect(topleft.x() * width, topleft.y() * height, sizes.x() * width, sizes.y() * height);

        // draw rectangle
        cv::rectangle(outImage.image, rect, color, 3);

        // draw label and score
        std::stringstream ss;
        ss << result.label << "(" << std::setprecision(2) << result.score << ")";
        cv::putText(
            outImage.image,
            ss.str(),
            cv::Point(
                std::min(static_cast<int>(topleft.x() * width), static_cast<int>(msg->width - 100)),
                std::max(static_cast<int>(topleft.y() * height - 10), 20)),
            cv::FONT_HERSHEY_PLAIN, 1.0, color);


        if (always_output_image_ || results.size() > 0)
        {
            imagePublisher_.publish(outImage.toImageMsg());
        }
    }

    // Get GP
    Eigen::Vector3d GPN(gp->n.data());
    double GPd =  gp->d * (-1000.0);
    Eigen::Matrix3d K(camera_info->K.data());

    if(pubDetectedPersons.getNumSubscribers())
    {
        spencer_tracking_msgs::DetectedPersons detected_persons;
        detected_persons.header = msg->header;

        for(const auto& result : results)
        {
            if (result.label_index != 1)
                continue;

            auto topleft = result.box.min();
            auto bbcenter = result.box.center();
            auto sizes = result.box.sizes();
            float width = sizes.x() * width * world_scale;
            float height = sizes.y() * height * world_scale;
            float x = topleft.x() + 16. * world_scale;
            float y = topleft.y() + 16. * world_scale;

            Eigen::Vector3d normal = Eigen::Vector3d();
            normal(0) = GPN(0);
            normal(1) = GPN(1);
            normal(2) = GPN(2);

            Eigen::Vector3d pos3D;
            calc3DPosFromBBox(K, normal, GPd, x, y, width, height, world_scale, pos3D);

            // DetectedPerson for SPENCER
            spencer_tracking_msgs::DetectedPerson detected_person;
            detected_person.modality = spencer_tracking_msgs::DetectedPerson::MODALITY_GENERIC_MONOCULAR_VISION;
            detected_person.confidence = result.score;
            detected_person.pose.pose.position.x = -pos3D(0) / 1000.;
            detected_person.pose.pose.position.y = -pos3D(1) / 1000.;
            detected_person.pose.pose.position.z = -pos3D(2) / 1000.;  
            detected_person.pose.pose.orientation.w = 1.0;

            const double LARGE_VARIANCE = 999999999;
            detected_person.pose.covariance[0*6 + 0] = pose_variance;
            detected_person.pose.covariance[1*6 + 1] = pose_variance; // up axis (since this is in sensor frame!)
            detected_person.pose.covariance[2*6 + 2] = pose_variance;
            detected_person.pose.covariance[3*6 + 3] = LARGE_VARIANCE;
            detected_person.pose.covariance[4*6 + 4] = LARGE_VARIANCE;
            detected_person.pose.covariance[5*6 + 5] = LARGE_VARIANCE;

            detected_person.detection_id = current_detection_id;
            current_detection_id += detection_id_increment;

            detected_persons.detections.push_back(detected_person);  
        }

        // Publish
        if (detected_persons.detections.size() > 0)
            pubDetectedPersons.publish(detected_persons);
    }
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "tensorflow_object_detector");

    TensorFlowObjectDetectorGPCore node(ros::NodeHandle(), ros::NodeHandle("~"));
    node.run();
    return 0;
}
