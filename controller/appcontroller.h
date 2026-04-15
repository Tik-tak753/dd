#ifndef APPCONTROLLER_H
#define APPCONTROLLER_H

#include <QImage>
#include <QString>

#include <memory>

#include <opencv2/videoio.hpp>

class IDetector;
class YoloDetector;

class AppController
{
public:
    AppController();
    ~AppController();

    bool loadOnnxModel(const QString &modelPath, QString *statusMessage);
    QString currentDetectorStatus() const;
    bool loadImageAndRunDetection(const QString &filePath, QImage *outputImage, QString *statusMessage) const;
    bool openVideo(const QString &filePath, QString *statusMessage);
    bool processNextVideoFrame(QImage *outputImage, QString *statusMessage, bool *hasFrame);
    void stopVideo();
    bool hasOpenVideo() const;

private:
    void activateStubDetector(const QString &reason);
    void activateYoloDetector(std::unique_ptr<YoloDetector> yoloDetector, const QString &modelPath);
    QString detectorStatusText() const;

    std::unique_ptr<IDetector> detector_;
    QString activeDetectorName_;
    QString loadedModelPath_;
    QString detectorStatusDetail_;
    QString loadedVideoPath_;
    cv::VideoCapture videoCapture_;
};

#endif // APPCONTROLLER_H
