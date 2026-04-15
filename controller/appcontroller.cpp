#include "appcontroller.h"

#include "detection/idetector.h"
#include "detection/stubdetector.h"
#include "utils/cvqtutils.h"

#include <QFile>

#include <opencv2/imgcodecs.hpp>

AppController::AppController()
    : detector_(std::make_unique<StubDetector>())
{
}

AppController::~AppController() = default;

bool AppController::loadImageAndRunDetection(const QString &filePath,
                                             QImage *outputImage,
                                             QString *statusMessage) const
{
    if (outputImage == nullptr || statusMessage == nullptr) {
        return false;
    }

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        *statusMessage = QStringLiteral("Failed to open image file: %1").arg(filePath);
        *outputImage = QImage();
        return false;
    }

    const QByteArray bytes = file.readAll();
    const std::vector<uchar> buffer(bytes.begin(), bytes.end());

    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);

    if (image.empty()) {
        *statusMessage = QStringLiteral("Failed to decode image: %1").arg(filePath);
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
