#ifndef YOLODETECTOR_H
#define YOLODETECTOR_H

#include "idetector.h"

#include <QString>

class YoloDetector : public IDetector
{
public:
    explicit YoloDetector(QString modelPath);

    DetectionList detect(const cv::Mat &image) const override;
    std::string detectorName() const override;

    bool isReady() const;
    QString modelPath() const;

private:
    QString modelPath_;
    bool ready_ = false;
};

#endif // YOLODETECTOR_H
