#include "appcontroller.h"

#include "detection/idetector.h"
#include "detection/stubdetector.h"
#include "utils/cvqtutils.h"

#include <opencv2/imgcodecs.hpp>

AppController::AppController()
    : detector_(std::make_unique<StubDetector>())
{
}

bool AppController::loadImageAndRunDetection(const QString &filePath,
                                             QImage *outputImage,
                                             QString *statusMessage) const
{
    if (outputImage == nullptr || statusMessage == nullptr) {
        return false;
    }

    const std::string filePathStd = filePath.toStdString();
    cv::Mat image = cv::imread(filePathStd, cv::IMREAD_COLOR);

    if (image.empty()) {
        *statusMessage = QStringLiteral("Failed to load image: %1").arg(filePath);
        *outputImage = QImage();
        return false;
    }

    const DetectionList detections = detector_->detect(image);
    CvQtUtils::drawDetections(image, detections);

    *outputImage = CvQtUtils::matToQImage(image);
    if (outputImage->isNull()) {
        *statusMessage = QStringLiteral("Loaded image, but conversion to QImage failed.");
        return false;
    }

    *statusMessage = QStringLiteral("Loaded %1 with %2 stub detection(s).")
                         .arg(filePath)
                         .arg(static_cast<int>(detections.size()));
    return true;
}
