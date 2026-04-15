#ifndef DETECTION_H
#define DETECTION_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct Detection
{
    cv::Rect box;
    float score = 0.0f;
    std::string label;
};

using DetectionList = std::vector<Detection>;

#endif // DETECTION_H
