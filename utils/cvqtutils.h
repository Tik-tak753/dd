#ifndef CVQTUTILS_H
#define CVQTUTILS_H

#include "detection/detection.h"

#include <QImage>

#include <opencv2/core.hpp>

namespace CvQtUtils {

void drawDetections(cv::Mat &image, const DetectionList &detections);
QImage matToQImage(const cv::Mat &image);

} // namespace CvQtUtils

#endif // CVQTUTILS_H
