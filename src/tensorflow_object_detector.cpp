#include "tensorflow_object_detector.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

TensorFlowObjectDetector::TensorFlowObjectDetector(const std::string& graph_path, const std::string& labels_path)
{
    //load graph
    {
        auto status = TensorFlowUtil::createStatus();
        graph_ = TensorFlowUtil::createGraph();

        //import
        auto graph_def = TensorFlowUtil::createBuffer(graph_path);
        auto options = TensorFlowUtil::createImportGraphDefOptions();
        TF_GraphImportGraphDef(graph_.get(), graph_def.get(), options.get(), status.get());
        TensorFlowUtil::throwIfError(status.get(), "Failed to load graph");
    }

    //create session
    {
        auto status = TensorFlowUtil::createStatus();
        auto options = TensorFlowUtil::createSessionOptions();
        uint8_t intra_op_parallelism_threads = 2;
        uint8_t inter_op_parallelism_threads = 1;
        uint8_t buf[]={0x10, intra_op_parallelism_threads, 0x28, inter_op_parallelism_threads};
        TF_SetConfig(options.get(), buf, sizeof(buf), status.get());
        if (TF_GetCode(status.get()) != TF_OK)
            std::cerr << "ERROR: " << TF_Message(status.get()) << std::endl;
        session_ = TensorFlowUtil::createSession(graph_.get(), options.get());
    }

    //read labels
    {
        std::ifstream file(labels_path);
        if (!file) {
            std::stringstream ss;
            ss << "Labels file " << labels_path << " is not found." << std::endl;
            throw std::invalid_argument(ss.str());
        }
        std::string line;
        while (std::getline(file, line)) {
            labels_.push_back(line);
        }
    }

    //setup operations
    {
        image_tensor_      = { TF_GraphOperationByName(graph_.get(), IMAGE_TENSOR.c_str())     , 0 };
        detection_boxes_   = { TF_GraphOperationByName(graph_.get(), DETECTION_BOXES.c_str())  , 0 };
        detection_scores_  = { TF_GraphOperationByName(graph_.get(), DETECTION_SCORES.c_str()) , 0 };
        detection_classes_ = { TF_GraphOperationByName(graph_.get(), DETECTION_CLASSES.c_str()), 0 };
        num_detections_    = { TF_GraphOperationByName(graph_.get(), NUM_DETECTIONS.c_str())   , 0 };
    }
}

std::vector<TensorFlowObjectDetector::Result> TensorFlowObjectDetector::detect(const cv::Mat& image, float score_threshold)
{
    const auto rows     = image.rows;
    const auto cols     = image.cols;
    const auto channels = image.channels();

    //setup input image tensor
    Eigen::TensorMap<Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>> eigen_tensor(image.data, 1, rows, cols, channels);

    auto image_tensor_value = TensorFlowUtil::createTensor<TF_UINT8, uint8_t, 4>(eigen_tensor);

    //run session
    std::array<std::unique_ptr<TF_Tensor>, 4> outputs;
    {
        auto status = TensorFlowUtil::createStatus();

        std::array<TF_Output , 1> input_ops    = { image_tensor_ };
        std::array<TF_Tensor*, 1> input_values = { image_tensor_value.get() };
        std::array<TF_Output , 4> output_ops   = { detection_boxes_, detection_scores_, detection_classes_, num_detections_ };

        std::array<TF_Tensor*, 4> output_values;
        TF_SessionRun(
            session_.get(), 
            nullptr,    //run options
            input_ops.data() , input_values.data() , input_ops.size(),
            output_ops.data(), output_values.data(), output_ops.size(),
            nullptr, 0, //targets
            nullptr,    //run metadata
            status.get()
        );

        for (int i = 0; i < outputs.size(); ++i)
        {
            outputs[i] = std::unique_ptr<TF_Tensor>(output_values[i]);
        }

        TensorFlowUtil::throwIfError(status.get(), "Failed to run session");
    }

    //copy results
    std::vector<Result> results;
    {
        const auto boxes_tensor          = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>(static_cast<float*>(TF_TensorData(outputs[0].get())), {100, 4});
        const auto scores_tensor         = Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>>(static_cast<float*>(TF_TensorData(outputs[1].get())), {100});
        const auto classes_tensor        = Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>>(static_cast<float*>(TF_TensorData(outputs[2].get())), {100});
        const auto num_detections_tensor = Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>>(static_cast<float*>(TF_TensorData(outputs[3].get())), {1});

        //retrieve and format valid results
        for(int i = 0; i < num_detections_tensor(0); ++i) {
            const float score = scores_tensor(i);
            if (score < score_threshold) {
                continue;
            }

            const Eigen::AlignedBox2f box(
                Eigen::Vector2f(boxes_tensor(i, 1), boxes_tensor(i, 0)),
                Eigen::Vector2f(boxes_tensor(i, 3), boxes_tensor(i, 2))
            );
            const int label_index = classes_tensor(i);

            std::string label;
            if (label_index <= labels_.size()) {
                label = labels_[label_index - 1];
            } else {
                label = "unknown";
            }
            
            std::stringstream ss;
            ss << classes_tensor(i) << " : " << label;

            results.push_back({
                box, score, label_index, ss.str()
            });
        }
    }

    return results;
}
