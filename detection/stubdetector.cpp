#include "stubdetector.h"

#include <algorithm>

DetectionList StubDetector::detect(const cv::Mat &image) const
{
    DetectionList detections;

    if (image.empty()) {
        return detections;
    }

    const int boxWidth = std::max(40, image.cols / 3);
    const int boxHeight = std::max(40, image.rows / 3);
    const int x = std::max(0, (image.cols - boxWidth) / 2);
    const int y = std::max(0, (image.rows - boxHeight) / 2);

    detections.push_back(Detection{cv::Rect(x, y, boxWidth, boxHeight), 0.85f, "stub-drone"});
    return detections;
}

std::string StubDetector::detectorName() const
{
    return "StubDetector";
}
