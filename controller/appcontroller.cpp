#include "appcontroller.h"

#include "detection/idetector.h"
#include "detection/stubdetector.h"
#include "detection/yolodetector.h"
#include "utils/cvqtutils.h"

#include <QFile>

#include <opencv2/imgcodecs.hpp>

AppController::AppController()
{
    activateStubDetector(QStringLiteral("no ONNX model loaded"));
}

AppController::~AppController() = default;

bool AppController::loadOnnxModel(const QString &modelPath, QString *statusMessage)
{
    if (statusMessage == nullptr) {
        return false;
    }

    const QString trimmedPath = modelPath.trimmed();
    if (trimmedPath.isEmpty()) {
        activateStubDetector(QStringLiteral("empty model path"));
        *statusMessage = QStringLiteral("[%1] Model load canceled: empty path selected.")
                             .arg(detectorStatusText());
        return false;
    }

    std::unique_ptr<YoloDetector> yoloDetector = std::make_unique<YoloDetector>(trimmedPath);
    if (yoloDetector->isReady()) {
        activateYoloDetector(std::move(yoloDetector), trimmedPath);
        *statusMessage = QStringLiteral("[%1] ONNX model loaded successfully.")
                             .arg(detectorStatusText());
        return true;
    }

    const QString yoloFailureDetail = yoloDetector->statusDetail();
    activateStubDetector(QStringLiteral("model load failed for %1 (%2)")
                             .arg(trimmedPath, yoloFailureDetail));
    *statusMessage = QStringLiteral("[%1] Failed to activate YoloDetector.")
                         .arg(detectorStatusText());
    return false;
}

QString AppController::currentDetectorStatus() const
{
    return detectorStatusText();
}

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
                             .arg(detectorStatusText(), filePath);
        *outputImage = QImage();
        return false;
    }

    const QByteArray bytes = file.readAll();
    const std::vector<uchar> buffer(bytes.begin(), bytes.end());

    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);

    if (image.empty()) {
        *statusMessage = QStringLiteral("[%1] Failed to decode image: %2")
                             .arg(detectorStatusText(), filePath);
        *outputImage = QImage();
        return false;
    }

    const DetectionList detections = detector_->detect(image);
    CvQtUtils::drawDetections(image, detections);

    *outputImage = CvQtUtils::matToQImage(image);
    if (outputImage->isNull()) {
        *statusMessage = QStringLiteral("[%1] Loaded image, but conversion to QImage failed.")
                             .arg(detectorStatusText());
        return false;
    }

    *statusMessage = QStringLiteral("[%1] Loaded %2 with %3 detection(s).")
                         .arg(detectorStatusText())
                         .arg(filePath)
                         .arg(static_cast<int>(detections.size()));
    return true;
}

void AppController::activateStubDetector(const QString &reason)
{
    detector_ = std::make_unique<StubDetector>();
    activeDetectorName_ = QStringLiteral("StubDetector");
    loadedModelPath_.clear();
    detectorStatusDetail_ = reason;
}

void AppController::activateYoloDetector(std::unique_ptr<YoloDetector> yoloDetector, const QString &modelPath)
{
    detector_ = std::move(yoloDetector);
    activeDetectorName_ = QStringLiteral("YoloDetector");
    loadedModelPath_ = modelPath;
    detectorStatusDetail_ = QStringLiteral("model loaded");
}

QString AppController::detectorStatusText() const
{
    const QString modelText = loadedModelPath_.isEmpty() ? QStringLiteral("<none>") : loadedModelPath_;
    return QStringLiteral("Detector=%1 | Model=%2 | Detail=%3")
        .arg(activeDetectorName_, modelText, detectorStatusDetail_);
}
