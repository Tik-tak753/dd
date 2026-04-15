#ifndef VIDEOINFERENCEWORKER_H
#define VIDEOINFERENCEWORKER_H

#include <QObject>
#include <QImage>
#include <QString>

class AppController;

class VideoInferenceWorker : public QObject
{
    Q_OBJECT

public:
    explicit VideoInferenceWorker(AppController *controller, QObject *parent = nullptr);

public slots:
    void processFrame(int generation);

signals:
    void frameProcessed(int generation,
                        const QImage &frameImage,
                        const QString &statusMessage,
                        bool hasFrame,
                        bool ok);

private:
    AppController *controller_;
};

#endif // VIDEOINFERENCEWORKER_H
