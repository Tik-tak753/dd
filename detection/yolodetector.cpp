#include "yolodetector.h"

#include <QFileInfo>
#include <QDebug>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

YoloDetector::YoloDetector(QString modelPath)
    : modelPath_(std::move(modelPath))
{
    const QFileInfo modelFile(modelPath_);
    if (!modelFile.exists() || !modelFile.isFile()) {
        statusDetail_ = QStringLiteral("invalid model path: %1").arg(modelPath_);
        return;
    }

    try {
        net_ = cv::dnn::readNetFromONNX(modelPath_.toStdString());
        if (net_.empty()) {
            statusDetail_ = QStringLiteral("failed to load ONNX network: %1").arg(modelPath_);
            return;
        }

        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        ready_ = true;
        statusDetail_ = QStringLiteral("model loaded: %1").arg(modelPath_);
    } catch (const cv::Exception &e) {
        statusDetail_ = QStringLiteral("ONNX load error: %1").arg(QString::fromStdString(e.what()));
        qWarning() << "YoloDetector initialization failed:" << statusDetail_;
    }
}

DetectionList YoloDetector::detect(const cv::Mat &image) const
{
    DetectionList detections;

    if (!ready_ || image.empty()) {
        return detections;
    }

    try {
        LetterboxInfo letterboxInfo;
        cv::Mat blob = preprocessLetterbox(image, &letterboxInfo);
        if (blob.empty()) {
            qWarning() << "YoloDetector preprocessing failed.";
            return detections;
        }

        qDebug() << "YoloDetector input image size:" << image.cols << "x" << image.rows;
        qDebug() << "YoloDetector network input size:" << inputWidth_ << "x" << inputHeight_;
        qDebug() << "YoloDetector letterbox: used =" << letterboxInfo.used
                 << "scale =" << letterboxInfo.scale
                 << "padLeft =" << letterboxInfo.padLeft
                 << "padTop =" << letterboxInfo.padTop;

        net_.setInput(blob);

        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        if (outputs.empty()) {
            return detections;
        }

        for (const cv::Mat &output : outputs) {
            QStringList shapeParts;
            for (int i = 0; i < output.dims; ++i) {
                shapeParts << QString::number(output.size[i]);
            }
            qDebug() << "YoloDetector raw output tensor shape:" << ("[" + shapeParts.join(",") + "]");

            DetectionList decoded = decodeDetections(output, image.size(), letterboxInfo);
            detections.insert(detections.end(), decoded.begin(), decoded.end());
        }
    } catch (const cv::Exception &e) {
        qWarning() << "YoloDetector inference failed:" << e.what();
        return DetectionList();
    }

    return detections;
}

cv::Mat YoloDetector::preprocessLetterbox(const cv::Mat &image, LetterboxInfo *letterboxInfo) const
{
    if (image.empty() || letterboxInfo == nullptr) {
        return cv::Mat();
    }

    const float scale = std::min(static_cast<float>(inputWidth_) / static_cast<float>(image.cols),
                                 static_cast<float>(inputHeight_) / static_cast<float>(image.rows));
    const int resizedWidth = static_cast<int>(std::round(static_cast<float>(image.cols) * scale));
    const int resizedHeight = static_cast<int>(std::round(static_cast<float>(image.rows) * scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0, cv::INTER_LINEAR);

    const int padWidth = inputWidth_ - resizedWidth;
    const int padHeight = inputHeight_ - resizedHeight;
    const int padLeft = padWidth / 2;
    const int padRight = padWidth - padLeft;
    const int padTop = padHeight / 2;
    const int padBottom = padHeight - padTop;

    cv::Mat padded;
    cv::copyMakeBorder(resized,
                       padded,
                       padTop,
                       padBottom,
                       padLeft,
                       padRight,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(114.0, 114.0, 114.0));

    cv::Mat blob;
    cv::dnn::blobFromImage(padded,
                           blob,
                           1.0 / 255.0,
                           cv::Size(inputWidth_, inputHeight_),
                           cv::Scalar(),
                           true,
                           false);

    letterboxInfo->scale = scale;
    letterboxInfo->padLeft = padLeft;
    letterboxInfo->padTop = padTop;
    letterboxInfo->used = (padLeft != 0 || padTop != 0 || resizedWidth != inputWidth_ || resizedHeight != inputHeight_);

    return blob;
}

DetectionList YoloDetector::decodeDetections(const cv::Mat &output,
                                             const cv::Size &sourceSize,
                                             const LetterboxInfo &letterboxInfo) const
{
    DetectionList detections;

    cv::Mat parsed = output;

    if (parsed.dims == 3 && parsed.size[0] == 1) {
        const int dim1 = parsed.size[1];
        const int dim2 = parsed.size[2];
        parsed = (dim2 >= dim1) ? parsed.reshape(1, {dim1, dim2})
                                : parsed.reshape(1, {dim2, dim1});
    } else if (parsed.dims > 2) {
        parsed = parsed.reshape(1, parsed.total() / parsed.size[parsed.dims - 1]);
    }

    if (parsed.empty() || parsed.cols < 5) {
        return detections;
    }

    if (parsed.rows > 0 && parsed.cols > 0 && parsed.rows < parsed.cols && parsed.rows <= 128) {
        parsed = parsed.t();
    }

    if (parsed.type() != CV_32F) {
        parsed.convertTo(parsed, CV_32F);
    }

    const bool hasObjectness = parsed.cols >= 6;
    const int classOffset = hasObjectness ? 5 : 4;
    const int classCount = std::max(0, parsed.cols - classOffset);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    const int rawCandidates = parsed.rows;
    int thresholdedCandidates = 0;

    for (int row = 0; row < parsed.rows; ++row) {
        const float *data = parsed.ptr<float>(row);

        float centerX = data[0];
        float centerY = data[1];
        float width = data[2];
        float height = data[3];

        if (!std::isfinite(centerX) || !std::isfinite(centerY) ||
            !std::isfinite(width) || !std::isfinite(height) ||
            width <= 0.0f || height <= 0.0f) {
            continue;
        }

        float objectness = 1.0f;
        if (hasObjectness) {
            objectness = data[4];
            if (!std::isfinite(objectness) || objectness < confidenceThreshold_) {
                continue;
            }
        }

        int classId = 0;
        float classScore = objectness;
        if (classCount > 0) {
            const cv::Mat classScores(1, classCount, CV_32F, const_cast<float *>(data + classOffset));
            cv::Point maxClassId;
            double maxClassScore = 0.0;
            cv::minMaxLoc(classScores, nullptr, &maxClassScore, nullptr, &maxClassId);
            classId = maxClassId.x;
            classScore = hasObjectness ? objectness * static_cast<float>(maxClassScore)
                                       : static_cast<float>(maxClassScore);
        }

        const float score = classScore;
        if (score < scoreThreshold_) {
            continue;
        }

        ++thresholdedCandidates;

        const bool normalizedCoords = centerX <= 1.5f && centerY <= 1.5f && width <= 1.5f && height <= 1.5f;
        if (normalizedCoords) {
            centerX *= static_cast<float>(inputWidth_);
            centerY *= static_cast<float>(inputHeight_);
            width *= static_cast<float>(inputWidth_);
            height *= static_cast<float>(inputHeight_);
        }

        centerX = (centerX - static_cast<float>(letterboxInfo.padLeft)) / letterboxInfo.scale;
        centerY = (centerY - static_cast<float>(letterboxInfo.padTop)) / letterboxInfo.scale;
        width /= letterboxInfo.scale;
        height /= letterboxInfo.scale;

        const int left = std::max(0, static_cast<int>(std::round(centerX - 0.5f * width)));
        const int top = std::max(0, static_cast<int>(std::round(centerY - 0.5f * height)));
        const int right = std::min(sourceSize.width, static_cast<int>(std::round(centerX + 0.5f * width)));
        const int bottom = std::min(sourceSize.height, static_cast<int>(std::round(centerY + 0.5f * height)));
        const int boxWidth = right - left;
        const int boxHeight = bottom - top;

        if (boxWidth <= 0 || boxHeight <= 0) {
            continue;
        }

        boxes.emplace_back(left, top, boxWidth, boxHeight);
        scores.push_back(score);
        classIds.push_back(classId);
    }

    qDebug() << "YoloDetector candidates: raw =" << rawCandidates
             << ", after threshold =" << thresholdedCandidates;

    std::vector<int> keptIndices;
    cv::dnn::NMSBoxes(boxes, scores, scoreThreshold_, nmsThreshold_, keptIndices);

    detections.reserve(keptIndices.size());
    for (const int index : keptIndices) {
        Detection detection;
        detection.box = boxes[index];
        detection.score = scores[index];
        detection.label = QStringLiteral("class_%1").arg(classIds[index]).toStdString();
        detections.push_back(std::move(detection));
    }

    qDebug() << "YoloDetector detections after NMS:" << detections.size();

    return detections;
}

std::string YoloDetector::detectorName() const
{
    return "YoloDetector";
}

bool YoloDetector::isReady() const
{
    return ready_;
}

QString YoloDetector::modelPath() const
{
    return modelPath_;
}

QString YoloDetector::statusDetail() const
{
    return statusDetail_;
}
