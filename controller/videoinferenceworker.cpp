#include "videoinferenceworker.h"

#include "appcontroller.h"

#include <QImage>
#include <QString>

VideoInferenceWorker::VideoInferenceWorker(AppController *controller, QObject *parent)
    : QObject(parent)
    , controller_(controller)
{
}

void VideoInferenceWorker::processFrame(int generation)
{
    QImage frameImage;
    QString statusMessage;
    bool hasFrame = false;
    const bool ok = controller_->processNextVideoFrame(&frameImage, &statusMessage, &hasFrame);

    emit frameProcessed(generation, frameImage, statusMessage, hasFrame, ok);
}
