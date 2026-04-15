#ifndef IDETECTOR_H
#define IDETECTOR_H

#include "detection.h"

#include <opencv2/core.hpp>

class IDetector
{
public:
    virtual ~IDetector() = default;
    virtual DetectionList detect(const cv::Mat &image) const = 0;
    virtual std::string detectorName() const = 0;
};

#endif // IDETECTOR_H
