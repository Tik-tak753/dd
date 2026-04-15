#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QTimer>

#include <memory>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class AppController;
class VideoInferenceWorker;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onOpenImageClicked();
    void onOpenVideoClicked();
    void onTogglePlaybackClicked();
    void onStopPlaybackClicked();
    void onLoadModelClicked();
    void onPerformanceProfileChanged(int index);
    void onPlaybackTick();
    void onFrameProcessed(int generation,
                          const QImage &frameImage,
                          const QString &statusMessage,
                          bool hasFrame,
                          bool ok);

private:
    void displayImage(const QImage &image);
    void setPlaybackRunning(bool isRunning);
    void requestNextVideoFrame();

    Ui::MainWindow *ui;
    std::unique_ptr<AppController> controller_;
    QTimer playbackTimer_;
    QThread videoWorkerThread_;
    VideoInferenceWorker *videoWorker_ = nullptr;
    bool isPlaybackRunning_ = false;
    bool inferenceInFlight_ = false;
    bool pendingInferenceRequest_ = false;
    int playbackGeneration_ = 0;
};

#endif // MAINWINDOW_H
