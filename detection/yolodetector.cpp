#include "yolodetector.h"

#include <QFileInfo>

#include <utility>

YoloDetector::YoloDetector(QString modelPath)
    : modelPath_(std::move(modelPath))
{
    const QFileInfo modelFile(modelPath_);
    ready_ = modelFile.exists() && modelFile.isFile();
}

DetectionList YoloDetector::detect(const cv::Mat &image) const
{
    DetectionList detections;

    if (!ready_ || image.empty()) {
        return detections;
    }

    // Placeholder for real ONNX inference integration.
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
