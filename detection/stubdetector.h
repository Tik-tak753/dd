#ifndef STUBDETECTOR_H
#define STUBDETECTOR_H

#include "idetector.h"

class StubDetector : public IDetector
{
public:
    DetectionList detect(const cv::Mat &image) const override;
    std::string detectorName() const override;
};

#endif // STUBDETECTOR_H
