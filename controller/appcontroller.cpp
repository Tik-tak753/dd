#include "appcontroller.h"

#include "detection/idetector.h"
#include "detection/stubdetector.h"
#include "detection/yolodetector.h"
#include "utils/cvqtutils.h"

#include <QFile>
#include <QProcessEnvironment>

#include <opencv2/imgcodecs.hpp>

namespace {

QString configuredYoloModelPath()
{
    return QProcessEnvironment::systemEnvironment().value(QStringLiteral("DRONE_YOLO_ONNX_PATH")).trimmed();
}

} // namespace

AppController::AppController()
{
    const QString yoloModelPath = configuredYoloModelPath();

    if (!yoloModelPath.isEmpty()) {
        std::unique_ptr<YoloDetector> yoloDetector = std::make_unique<YoloDetector>(yoloModelPath);
        if (yoloDetector->isReady()) {
            detector_ = std::move(yoloDetector);
            detectorStatusDetail_ = QStringLiteral("YoloDetector active (model: %1)").arg(yoloModelPath);
        } else {
            detectorStatusDetail_ = QStringLiteral("YoloDetector unavailable (%1)").arg(yoloDetector->statusDetail());
        }
    }

    if (!detector_) {
        detector_ = std::make_unique<StubDetector>();
        if (yoloModelPath.isEmpty()) {
            detectorStatusDetail_ = QStringLiteral(
                "StubDetector active (fallback: DRONE_YOLO_ONNX_PATH is not configured)");
        } else if (!detectorStatusDetail_.isEmpty()) {
            detectorStatusDetail_ = QStringLiteral("StubDetector active (fallback: %1)").arg(detectorStatusDetail_);
        } else {
            detectorStatusDetail_ = QStringLiteral("StubDetector active (fallback: invalid YOLO setup)");
        }
    }
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
        *statusMessage = QStringLiteral("[%1] Failed to open image file: %2")
                             .arg(detectorStatusDetail_, filePath);
        *outputImage = QImage();
        return false;
    }

    const QByteArray bytes = file.readAll();
    const std::vector<uchar> buffer(bytes.begin(), bytes.end());

    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);

    if (image.empty()) {
        *statusMessage = QStringLiteral("[%1] Failed to decode image: %2")
                             .arg(detectorStatusDetail_, filePath);
        *outputImage = QImage();
        return false;
    }

    const DetectionList detections = detector_->detect(image);
    CvQtUtils::drawDetections(image, detections);

    *outputImage = CvQtUtils::matToQImage(image);
    if (outputImage->isNull()) {
        *statusMessage = QStringLiteral("[%1] Loaded image, but conversion to QImage failed.")
                             .arg(detectorStatusDetail_);
        return false;
    }

    *statusMessage = QStringLiteral("[%1] Loaded %2 with %3 detection(s).")
                         .arg(detectorStatusDetail_)
                         .arg(filePath)
                         .arg(static_cast<int>(detections.size()));
    return true;
}
