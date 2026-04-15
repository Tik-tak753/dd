#include "appcontroller.h"

#include "detection/idetector.h"
#include "detection/stubdetector.h"
#include "detection/yolodetector.h"
#include "utils/cvqtutils.h"

#include <QDebug>
#include <QFile>

#include <algorithm>
#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

AppController::AppController()
{
    activateStubDetector(QStringLiteral("no ONNX model loaded"));
    applyVideoPerformanceSettings(settingsForProfile(videoPerformanceProfile_));
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

bool AppController::openVideo(const QString &filePath, QString *statusMessage)
{
    if (statusMessage == nullptr) {
        return false;
    }

    const QString trimmedPath = filePath.trimmed();
    if (trimmedPath.isEmpty()) {
        *statusMessage = QStringLiteral("[%1] Failed to open video: empty path.")
                             .arg(detectorStatusText());
        return false;
    }

    cv::VideoCapture capture(trimmedPath.toStdString());
    if (!capture.isOpened()) {
        *statusMessage = QStringLiteral("[%1] Failed to open video file: %2")
                             .arg(detectorStatusText(), trimmedPath);
        return false;
    }

    videoCapture_ = std::move(capture);
    loadedVideoPath_ = trimmedPath;
    videoFrameIndex_ = 0;
    cachedVideoDetections_.clear();
    hasLastFrameTimestamp_ = false;
    smoothedFps_ = 0.0;
    *statusMessage = QStringLiteral("[%1] Video opened: %2 | Profile=%3 | Mode=%4")
                         .arg(detectorStatusText())
                         .arg(loadedVideoPath_)
                         .arg(currentVideoPerformanceProfileName())
                         .arg(videoInferenceModeText());
    return true;
}

bool AppController::processNextVideoFrame(QImage *outputImage,
                                          QString *statusMessage,
                                          bool *hasFrame)
{
    if (outputImage == nullptr || statusMessage == nullptr || hasFrame == nullptr) {
        return false;
    }

    if (!videoCapture_.isOpened()) {
        *hasFrame = false;
        *statusMessage = QStringLiteral("[%1] No video is currently open.")
                             .arg(detectorStatusText());
        *outputImage = QImage();
        return false;
    }

    cv::Mat frame;
    if (!videoCapture_.read(frame) || frame.empty()) {
        *hasFrame = false;
        *outputImage = QImage();
        *statusMessage = QStringLiteral("[%1] End of video: %2")
                             .arg(detectorStatusText(), loadedVideoPath_);
        return true;
    }

    const auto now = std::chrono::steady_clock::now();
    if (hasLastFrameTimestamp_) {
        const auto elapsedMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFrameTimestamp_).count();
        if (elapsedMs > 0) {
            const double instantFps = 1000.0 / static_cast<double>(elapsedMs);
            if (smoothedFps_ <= 0.0) {
                smoothedFps_ = instantFps;
            } else {
                smoothedFps_ = smoothedFps_ * 0.85 + instantFps * 0.15;
            }
        }
    }
    lastFrameTimestamp_ = now;
    hasLastFrameTimestamp_ = true;

    DetectionList detectionsToDraw = cachedVideoDetections_;
    const bool shouldRunDetection =
        cachedVideoDetections_.empty() || videoInferenceFrameSkip_ <= 0 ||
        (videoFrameIndex_ % (videoInferenceFrameSkip_ + 1) == 0);
    bool ranDetection = false;
    cv::Size inferenceFrameSize = frame.size();

    if (shouldRunDetection) {
        cv::Mat inferenceFrame = createVideoInferenceFrame(frame);
        inferenceFrameSize = inferenceFrame.size();
        DetectionList detections = detector_->detect(inferenceFrame);
        detectionsToDraw = remapDetectionsToDisplayFrame(detections, inferenceFrame.size(), frame.size());
        cachedVideoDetections_ = detectionsToDraw;
        ranDetection = true;
    } else if (reduceVideoInferenceResolution_) {
        const int targetWidth = std::max(1, videoInferenceResolution_.width);
        const int targetHeight = std::max(1, videoInferenceResolution_.height);
        if (frame.cols > targetWidth || frame.rows > targetHeight) {
            const float scale =
                std::min(static_cast<float>(targetWidth) / static_cast<float>(frame.cols),
                         static_cast<float>(targetHeight) / static_cast<float>(frame.rows));
            inferenceFrameSize = cv::Size(
                std::max(1, static_cast<int>(std::round(static_cast<float>(frame.cols) * scale))),
                std::max(1, static_cast<int>(std::round(static_cast<float>(frame.rows) * scale))));
        }
    }

    const QString inferencePath = (inferenceFrameSize == frame.size())
        ? QStringLiteral("original-frame")
        : QStringLiteral("resized-copy");
    qInfo().noquote()
        << QStringLiteral(
               "VideoFrame[%1] Profile=%2 FrameSkip=%3 SourceSize=%4x%5 InferenceSize=%6x%7 InferencePath=%8 DetectorStep=%9")
               .arg(videoFrameIndex_)
               .arg(currentVideoPerformanceProfileName())
               .arg(videoInferenceFrameSkip_)
               .arg(frame.cols)
               .arg(frame.rows)
               .arg(inferenceFrameSize.width)
               .arg(inferenceFrameSize.height)
               .arg(inferencePath)
               .arg(ranDetection ? QStringLiteral("run") : QStringLiteral("skip"));

    CvQtUtils::drawDetections(frame, detectionsToDraw);

    *outputImage = CvQtUtils::matToQImage(frame);
    if (outputImage->isNull()) {
        *hasFrame = false;
        *statusMessage = QStringLiteral("[%1] Video frame conversion failed.")
                             .arg(detectorStatusText());
        return false;
    }

    *hasFrame = true;
    *statusMessage =
        QStringLiteral("[%1] Video=%2 | Detections=%3 | FPS=%4 | Profile=%5 | Mode=%6 | Inference=%7x%8 (%9) | FrameSkip=%10 | DetectorStep=%11")
                         .arg(detectorStatusText(), loadedVideoPath_)
                         .arg(static_cast<int>(detectionsToDraw.size()))
                         .arg(QString::number(smoothedFps_, 'f', 1))
                         .arg(currentVideoPerformanceProfileName())
                         .arg(videoInferenceModeText())
                         .arg(inferenceFrameSize.width)
                         .arg(inferenceFrameSize.height)
                         .arg(inferencePath)
                         .arg(videoInferenceFrameSkip_)
                         .arg(ranDetection ? QStringLiteral("run") : QStringLiteral("skip"));
    ++videoFrameIndex_;
    return true;
}

void AppController::stopVideo()
{
    if (videoCapture_.isOpened()) {
        videoCapture_.release();
    }
    loadedVideoPath_.clear();
    cachedVideoDetections_.clear();
    videoFrameIndex_ = 0;
    hasLastFrameTimestamp_ = false;
    smoothedFps_ = 0.0;
}

bool AppController::hasOpenVideo() const
{
    return videoCapture_.isOpened();
}

bool AppController::setVideoPerformanceProfile(VideoPerformanceProfile profile, QString *statusMessage)
{
    if (statusMessage == nullptr) {
        return false;
    }

    videoPerformanceProfile_ = profile;
    applyVideoPerformanceSettings(settingsForProfile(videoPerformanceProfile_));
    cachedVideoDetections_.clear();
    videoFrameIndex_ = 0;

    *statusMessage = QStringLiteral("[%1] Video profile set to %2 (%3)")
                         .arg(detectorStatusText())
                         .arg(currentVideoPerformanceProfileName())
                         .arg(videoInferenceModeText());
    return true;
}

QString AppController::currentVideoPerformanceProfileName() const
{
    return settingsForProfile(videoPerformanceProfile_).profileName;
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

AppController::VideoPerformanceSettings
AppController::settingsForProfile(VideoPerformanceProfile profile) const
{
    switch (profile) {
    case VideoPerformanceProfile::Fast:
        return {QStringLiteral("Fast"), true, cv::Size(416, 234), 2};
    case VideoPerformanceProfile::Accurate:
        return {QStringLiteral("Accurate"), true, cv::Size(960, 540), 0};
    case VideoPerformanceProfile::Balanced:
    default:
        return {QStringLiteral("Balanced"), true, cv::Size(640, 360), 1};
    }
}

void AppController::applyVideoPerformanceSettings(const VideoPerformanceSettings &settings)
{
    reduceVideoInferenceResolution_ = settings.reduceResolution;
    videoInferenceResolution_ = settings.inferenceResolution;
    videoInferenceFrameSkip_ = std::max(0, settings.frameSkip);
}

cv::Mat AppController::createVideoInferenceFrame(const cv::Mat &frame) const
{
    if (frame.empty() || !reduceVideoInferenceResolution_) {
        return frame;
    }

    const int targetWidth = std::max(1, videoInferenceResolution_.width);
    const int targetHeight = std::max(1, videoInferenceResolution_.height);
    if (frame.cols <= targetWidth && frame.rows <= targetHeight) {
        return frame;
    }

    const float scale = std::min(static_cast<float>(targetWidth) / static_cast<float>(frame.cols),
                                 static_cast<float>(targetHeight) / static_cast<float>(frame.rows));
    const int resizedWidth = std::max(1, static_cast<int>(std::round(static_cast<float>(frame.cols) * scale)));
    const int resizedHeight = std::max(1, static_cast<int>(std::round(static_cast<float>(frame.rows) * scale)));

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0, cv::INTER_LINEAR);
    return resized;
}

DetectionList AppController::remapDetectionsToDisplayFrame(const DetectionList &detections,
                                                           const cv::Size &inferenceSize,
                                                           const cv::Size &displaySize) const
{
    if (detections.empty() || inferenceSize.width <= 0 || inferenceSize.height <= 0 ||
        displaySize.width <= 0 || displaySize.height <= 0) {
        return detections;
    }

    const float scaleX = static_cast<float>(displaySize.width) / static_cast<float>(inferenceSize.width);
    const float scaleY = static_cast<float>(displaySize.height) / static_cast<float>(inferenceSize.height);

    DetectionList remapped = detections;
    for (Detection &detection : remapped) {
        const int left = std::max(0, static_cast<int>(std::round(static_cast<float>(detection.box.x) * scaleX)));
        const int top = std::max(0, static_cast<int>(std::round(static_cast<float>(detection.box.y) * scaleY)));
        const int width = std::max(0, static_cast<int>(std::round(static_cast<float>(detection.box.width) * scaleX)));
        const int height = std::max(0, static_cast<int>(std::round(static_cast<float>(detection.box.height) * scaleY)));

        const int boundedWidth = std::min(width, displaySize.width - left);
        const int boundedHeight = std::min(height, displaySize.height - top);
        detection.box = cv::Rect(left, top, std::max(0, boundedWidth), std::max(0, boundedHeight));
    }

    return remapped;
}

QString AppController::videoInferenceModeText() const
{
    return QStringLiteral("infer=%1@%2x%3,skip=%4")
        .arg(reduceVideoInferenceResolution_ ? QStringLiteral("resized") : QStringLiteral("full"))
        .arg(videoInferenceResolution_.width)
        .arg(videoInferenceResolution_.height)
        .arg(videoInferenceFrameSkip_);
}
