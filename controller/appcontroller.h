#ifndef APPCONTROLLER_H
#define APPCONTROLLER_H

#include "detection/detection.h"

#include <QImage>
#include <QString>

#include <chrono>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

class IDetector;
class YoloDetector;

class AppController
{
public:
    AppController();
    ~AppController();

    bool loadOnnxModel(const QString &modelPath, QString *statusMessage);
    QString currentDetectorStatus() const;
    bool loadImageAndRunDetection(const QString &filePath, QImage *outputImage, QString *statusMessage) const;
    bool openVideo(const QString &filePath, QString *statusMessage);
    bool processNextVideoFrame(QImage *outputImage, QString *statusMessage, bool *hasFrame);
    void stopVideo();
    bool hasOpenVideo() const;

private:
    cv::Mat createVideoInferenceFrame(const cv::Mat &frame) const;
    DetectionList remapDetectionsToDisplayFrame(const DetectionList &detections,
                                                const cv::Size &inferenceSize,
                                                const cv::Size &displaySize) const;
    QString videoInferenceModeText() const;

    void activateStubDetector(const QString &reason);
    void activateYoloDetector(std::unique_ptr<YoloDetector> yoloDetector, const QString &modelPath);
    QString detectorStatusText() const;

    std::unique_ptr<IDetector> detector_;
    QString activeDetectorName_;
    QString loadedModelPath_;
    QString detectorStatusDetail_;
    QString loadedVideoPath_;
    cv::VideoCapture videoCapture_;

    bool reduceVideoInferenceResolution_ = true;
    cv::Size videoInferenceResolution_ = cv::Size(640, 360);
    int videoInferenceFrameSkip_ = 1;
    int videoFrameIndex_ = 0;
    DetectionList cachedVideoDetections_;

    bool hasLastFrameTimestamp_ = false;
    std::chrono::steady_clock::time_point lastFrameTimestamp_;
    double smoothedFps_ = 0.0;
};

#endif // APPCONTROLLER_H
