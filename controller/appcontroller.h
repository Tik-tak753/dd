#ifndef APPCONTROLLER_H
#define APPCONTROLLER_H

#include <QImage>
#include <QString>

#include <memory>

class IDetector;

class AppController
{
public:
    AppController();
    ~AppController();

    bool loadImageAndRunDetection(const QString &filePath, QImage *outputImage, QString *statusMessage) const;

private:
    std::unique_ptr<IDetector> detector_;
    QString detectorStatusDetail_;
};

#endif // APPCONTROLLER_H
