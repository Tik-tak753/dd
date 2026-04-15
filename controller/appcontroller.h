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

    bool loadImageAndRunDetection(const QString &filePath, QImage *outputImage, QString *statusMessage) const;

private:
    std::unique_ptr<IDetector> detector_;
};

#endif // APPCONTROLLER_H
