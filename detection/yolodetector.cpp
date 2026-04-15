#include "yolodetector.h"

#include <QFileInfo>
#include <QDebug>

#include <algorithm>
#include <utility>
#include <vector>

#include <opencv2/dnn/dnn.hpp>

YoloDetector::YoloDetector(QString modelPath)
    : modelPath_(std::move(modelPath))
{
    const QFileInfo modelFile(modelPath_);
    if (!modelFile.exists() || !modelFile.isFile()) {
        statusDetail_ = QStringLiteral("invalid model path: %1").arg(modelPath_);
        return;
    }

    try {
        net_ = cv::dnn::readNetFromONNX(modelPath_.toStdString());
        if (net_.empty()) {
            statusDetail_ = QStringLiteral("failed to load ONNX network: %1").arg(modelPath_);
            return;
        }

        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        ready_ = true;
        statusDetail_ = QStringLiteral("model loaded: %1").arg(modelPath_);
    } catch (const cv::Exception &e) {
        statusDetail_ = QStringLiteral("ONNX load error: %1").arg(QString::fromStdString(e.what()));
        qWarning() << "YoloDetector initialization failed:" << statusDetail_;
    }
}

DetectionList YoloDetector::detect(const cv::Mat &image) const
{
    DetectionList detections;

    if (!ready_ || image.empty()) {
        return detections;
    }

    try {
        cv::Mat blob;
        cv::dnn::blobFromImage(image,
                               blob,
                               1.0 / 255.0,
                               cv::Size(inputWidth_, inputHeight_),
                               cv::Scalar(),
                               true,
                               false);

        net_.setInput(blob);

        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        if (outputs.empty()) {
            return detections;
        }

        for (const cv::Mat &output : outputs) {
            DetectionList decoded = decodeDetections(output, image.size());
            detections.insert(detections.end(), decoded.begin(), decoded.end());
        }
    } catch (const cv::Exception &e) {
        qWarning() << "YoloDetector inference failed:" << e.what();
        return DetectionList();
    }

    return detections;
}

DetectionList YoloDetector::decodeDetections(const cv::Mat &output, const cv::Size &sourceSize) const
{
    DetectionList detections;

    cv::Mat parsed = output;

    if (parsed.dims == 3 && parsed.size[0] == 1) {
        if (parsed.size[1] > parsed.size[2]) {
            parsed = parsed.reshape(1, {parsed.size[1], parsed.size[2]});
        } else {
            parsed = parsed.reshape(1, {parsed.size[2], parsed.size[1]});
        }
    } else if (parsed.dims > 2) {
        parsed = parsed.reshape(1, parsed.total() / parsed.size[parsed.dims - 1]);
    }

    if (parsed.empty() || parsed.cols < 5) {
        return detections;
    }

    if (parsed.type() != CV_32F) {
        parsed.convertTo(parsed, CV_32F);
    }

    const float scaleX = static_cast<float>(sourceSize.width) / static_cast<float>(inputWidth_);
    const float scaleY = static_cast<float>(sourceSize.height) / static_cast<float>(inputHeight_);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    for (int row = 0; row < parsed.rows; ++row) {
        const float *data = parsed.ptr<float>(row);

        const float centerX = data[0];
        const float centerY = data[1];
        const float width = data[2];
        const float height = data[3];

        float objectness = 1.0f;
        int classId = 0;
        float classScore = 1.0f;

        if (parsed.cols > 5) {
            objectness = data[4];
            if (objectness < confidenceThreshold_) {
                continue;
            }

            const cv::Mat classScores(1, parsed.cols - 5, CV_32F, const_cast<float *>(data + 5));
            cv::Point maxClassId;
            double maxClassScore = 0.0;
            cv::minMaxLoc(classScores, nullptr, &maxClassScore, nullptr, &maxClassId);
            classId = maxClassId.x;
            classScore = static_cast<float>(maxClassScore);
        } else {
            objectness = data[4];
        }

        const float score = objectness * classScore;
        if (score < scoreThreshold_) {
            continue;
        }

        const int left = std::max(0, static_cast<int>((centerX - 0.5f * width) * scaleX));
        const int top = std::max(0, static_cast<int>((centerY - 0.5f * height) * scaleY));
        const int boxWidth = std::min(sourceSize.width - left, static_cast<int>(width * scaleX));
        const int boxHeight = std::min(sourceSize.height - top, static_cast<int>(height * scaleY));

        if (boxWidth <= 0 || boxHeight <= 0) {
            continue;
        }

        boxes.emplace_back(left, top, boxWidth, boxHeight);
        scores.push_back(score);
        classIds.push_back(classId);
    }

    std::vector<int> keptIndices;
    cv::dnn::NMSBoxes(boxes, scores, scoreThreshold_, nmsThreshold_, keptIndices);

    detections.reserve(keptIndices.size());
    for (const int index : keptIndices) {
        Detection detection;
        detection.box = boxes[index];
        detection.score = scores[index];
        detection.label = QStringLiteral("class_%1").arg(classIds[index]).toStdString();
        detections.push_back(std::move(detection));
    }

    return detections;
}

std::string YoloDetector::detectorName() const
{
    return "YoloDetector";
}

bool YoloDetector::isReady() const
{
    return ready_;
}

QString YoloDetector::modelPath() const
{
    return modelPath_;
}

QString YoloDetector::statusDetail() const
{
    return statusDetail_;
}
