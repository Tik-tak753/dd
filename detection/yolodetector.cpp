#include "yolodetector.h"

#include <QFileInfo>
#include <QDebug>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <opencv2/dnn/dnn.hpp>

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
        cv::Mat blob;
        cv::dnn::blobFromImage(image,
                               blob,
                               1.0 / 255.0,
                               cv::Size(inputWidth_, inputHeight_),
                               cv::Scalar(),
                               true,
                               false);

        net_.setInput(blob);

        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        if (outputs.empty()) {
            return detections;
        }

        for (const cv::Mat &output : outputs) {
            DetectionList decoded = decodeDetections(output, image.size());
            detections.insert(detections.end(), decoded.begin(), decoded.end());
        }
    } catch (const cv::Exception &e) {
        qWarning() << "YoloDetector inference failed:" << e.what();
        return DetectionList();
    }

    return detections;
}

DetectionList YoloDetector::decodeDetections(const cv::Mat &output, const cv::Size &sourceSize) const
{
    DetectionList detections;

    QStringList shapeParts;
    for (int i = 0; i < output.dims; ++i) {
        shapeParts << QString::number(output.size[i]);
    }
    qDebug() << "YoloDetector output shape:" << ("[" + shapeParts.join(",") + "]");

    cv::Mat parsed = output;

    if (parsed.dims == 3 && parsed.size[0] == 1) {
        if (parsed.size[1] > parsed.size[2]) {
            parsed = parsed.reshape(1, {parsed.size[1], parsed.size[2]});
        } else {
            parsed = parsed.reshape(1, {parsed.size[2], parsed.size[1]});
        }
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

    const float inputScaleX = static_cast<float>(sourceSize.width) / static_cast<float>(inputWidth_);
    const float inputScaleY = static_cast<float>(sourceSize.height) / static_cast<float>(inputHeight_);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    const int rawCandidates = parsed.rows;
    int confidenceCandidates = 0;

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

        ++confidenceCandidates;

        const bool normalizedCoords =
            centerX <= 1.5f && centerY <= 1.5f && width <= 1.5f && height <= 1.5f;
        const float coordScaleX = normalizedCoords ? static_cast<float>(sourceSize.width) : inputScaleX;
        const float coordScaleY = normalizedCoords ? static_cast<float>(sourceSize.height) : inputScaleY;

        centerX *= coordScaleX;
        centerY *= coordScaleY;
        width *= coordScaleX;
        height *= coordScaleY;

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
             << ", after confidence =" << confidenceCandidates;

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
