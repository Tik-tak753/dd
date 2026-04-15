#include "cvqtutils.h"

#include <QString>

#include <opencv2/imgproc.hpp>

namespace CvQtUtils {

void drawDetections(cv::Mat &image, const DetectionList &detections)
{
    const cv::Scalar color(0, 255, 0);

    for (const Detection &detection : detections) {
        cv::rectangle(image, detection.box, color, 2);

        const std::string caption = detection.label + " " + std::to_string(detection.score);
        const cv::Point labelOrigin(detection.box.x, std::max(20, detection.box.y - 8));
        cv::putText(image, caption, labelOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }
}

QImage matToQImage(const cv::Mat &image)
{
    if (image.empty()) {
        return QImage();
    }

    if (image.type() == CV_8UC1) {
        return QImage(image.data,
                      image.cols,
                      image.rows,
                      static_cast<int>(image.step),
                      QImage::Format_Grayscale8)
            .copy();
    }

    if (image.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data,
                      rgb.cols,
                      rgb.rows,
                      static_cast<int>(rgb.step),
                      QImage::Format_RGB888)
            .copy();
    }

    if (image.type() == CV_8UC4) {
        return QImage(image.data,
                      image.cols,
                      image.rows,
                      static_cast<int>(image.step),
                      QImage::Format_RGBA8888)
            .copy();
    }

    return QImage();
}

} // namespace CvQtUtils
