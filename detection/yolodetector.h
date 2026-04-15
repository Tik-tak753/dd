#ifndef YOLODETECTOR_H
#define YOLODETECTOR_H

#include "idetector.h"

#include <QString>

#include <opencv2/dnn.hpp>

class YoloDetector : public IDetector
{
public:
    explicit YoloDetector(QString modelPath);

    DetectionList detect(const cv::Mat &image) const override;
    std::string detectorName() const override;

    bool isReady() const;
    QString modelPath() const;
    QString statusDetail() const;

private:
    struct LetterboxInfo
    {
        float scale = 1.0f;
        int padLeft = 0;
        int padTop = 0;
        bool used = false;
    };

    cv::Mat preprocessLetterbox(const cv::Mat &image, LetterboxInfo *letterboxInfo) const;
    DetectionList decodeDetections(const cv::Mat &output,
                                   const cv::Size &sourceSize,
                                   const LetterboxInfo &letterboxInfo) const;

    QString modelPath_;
    QString statusDetail_;
    mutable cv::dnn::Net net_;
    bool ready_ = false;

    int inputWidth_ = 640;
    int inputHeight_ = 640;
    float confidenceThreshold_ = 0.25f;
    float scoreThreshold_ = 0.25f;
    float nmsThreshold_ = 0.45f;
};

#endif // YOLODETECTOR_H
