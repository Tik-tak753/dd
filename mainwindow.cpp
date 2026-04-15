#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "controller/appcontroller.h"
#include "controller/videoinferenceworker.h"

#include <QFileDialog>
#include <QComboBox>
#include <QImage>
#include <QMetaObject>
#include <QPixmap>

namespace {
AppController::VideoPerformanceProfile profileFromIndex(int index)
{
    switch (index) {
    case 0:
        return AppController::VideoPerformanceProfile::Fast;
    case 2:
        return AppController::VideoPerformanceProfile::Accurate;
    case 1:
    default:
        return AppController::VideoPerformanceProfile::Balanced;
    }
}
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , controller_(std::make_unique<AppController>())
{
    ui->setupUi(this);
    setWindowTitle(QStringLiteral("Drone Detection Demo"));
    playbackTimer_.setInterval(33);

    videoWorker_ = new VideoInferenceWorker(controller_.get());
    videoWorker_->moveToThread(&videoWorkerThread_);
    connect(&videoWorkerThread_, &QThread::finished, videoWorker_, &QObject::deleteLater);
    connect(videoWorker_,
            &VideoInferenceWorker::frameProcessed,
            this,
            &MainWindow::onFrameProcessed,
            Qt::QueuedConnection);
    videoWorkerThread_.start();

    connect(ui->openImageButton, &QPushButton::clicked, this, &MainWindow::onOpenImageClicked);
    connect(ui->openVideoButton, &QPushButton::clicked, this, &MainWindow::onOpenVideoClicked);
    connect(ui->playPauseButton, &QPushButton::clicked, this, &MainWindow::onTogglePlaybackClicked);
    connect(ui->stopPlaybackButton, &QPushButton::clicked, this, &MainWindow::onStopPlaybackClicked);
    connect(ui->loadModelButton, &QPushButton::clicked, this, &MainWindow::onLoadModelClicked);
    connect(ui->performanceProfileComboBox,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            &MainWindow::onPerformanceProfileChanged);
    connect(&playbackTimer_, &QTimer::timeout, this, &MainWindow::onPlaybackTick);

    ui->performanceProfileComboBox->setCurrentIndex(1);
    QString profileStatusMessage;
    controller_->setVideoPerformanceProfile(profileFromIndex(ui->performanceProfileComboBox->currentIndex()),
                                            &profileStatusMessage);

    setPlaybackRunning(false);
    statusBar()->showMessage(
        QStringLiteral("Ready. %1 | Profile=%2")
            .arg(controller_->currentDetectorStatus(), controller_->currentVideoPerformanceProfileName()));
}

MainWindow::~MainWindow()
{
    setPlaybackRunning(false);
    pendingInferenceRequest_ = false;
    videoWorkerThread_.quit();
    videoWorkerThread_.wait();
    delete ui;
}

void MainWindow::onOpenImageClicked()
{
    const QString filePath = QFileDialog::getOpenFileName(this,
                                                          QStringLiteral("Open Image"),
                                                          QString(),
                                                          QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp)"));
    if (filePath.isEmpty()) {
        return;
    }

    QImage outputImage;
    QString statusMessage;
    const bool ok = controller_->loadImageAndRunDetection(filePath, &outputImage, &statusMessage);

    if (ok) {
        displayImage(outputImage);
    }

    statusBar()->showMessage(statusMessage);
}

void MainWindow::onLoadModelClicked()
{
    const QString modelPath = QFileDialog::getOpenFileName(this,
                                                           QStringLiteral("Load ONNX Model"),
                                                           QString(),
                                                           QStringLiteral("ONNX Models (*.onnx)"));
    if (modelPath.isEmpty()) {
        statusBar()->showMessage(QStringLiteral("Model selection canceled. %1")
                                     .arg(controller_->currentDetectorStatus()));
        return;
    }

    QString statusMessage;
    controller_->loadOnnxModel(modelPath, &statusMessage);
    statusBar()->showMessage(statusMessage);
}

void MainWindow::onPerformanceProfileChanged(int index)
{
    QString statusMessage;
    controller_->setVideoPerformanceProfile(profileFromIndex(index), &statusMessage);
    statusBar()->showMessage(statusMessage);
}

void MainWindow::onOpenVideoClicked()
{
    const QString filePath = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("Open Video"),
        QString(),
        QStringLiteral("Videos (*.mp4 *.avi *.mov *.mkv *.wmv)"));
    if (filePath.isEmpty()) {
        return;
    }

    QString statusMessage;
    ++playbackGeneration_;
    inferenceInFlight_ = false;
    pendingInferenceRequest_ = false;

    if (controller_->openVideo(filePath, &statusMessage)) {
        setPlaybackRunning(true);
        requestNextVideoFrame();
    } else {
        setPlaybackRunning(false);
    }

    statusBar()->showMessage(statusMessage);
}

void MainWindow::onTogglePlaybackClicked()
{
    if (!controller_->hasOpenVideo()) {
        statusBar()->showMessage(QStringLiteral("No video loaded. Open a video file first."));
        return;
    }

    setPlaybackRunning(!isPlaybackRunning_);
    statusBar()->showMessage(isPlaybackRunning_
                                 ? QStringLiteral("Playback started.")
                                 : QStringLiteral("Playback paused."));
}

void MainWindow::onStopPlaybackClicked()
{
    setPlaybackRunning(false);
    ++playbackGeneration_;
    pendingInferenceRequest_ = false;
    controller_->stopVideo();
    statusBar()->showMessage(QStringLiteral("Playback stopped."));
}

void MainWindow::onPlaybackTick()
{
    requestNextVideoFrame();
}

void MainWindow::onFrameProcessed(int generation,
                                  const QImage &frameImage,
                                  const QString &statusMessage,
                                  bool hasFrame,
                                  bool ok)
{
    inferenceInFlight_ = false;

    if (generation != playbackGeneration_) {
        return;
    }

    if (ok && hasFrame) {
        displayImage(frameImage);
    } else {
        setPlaybackRunning(false);
        pendingInferenceRequest_ = false;
    }

    statusBar()->showMessage(statusMessage);

    if (isPlaybackRunning_ && pendingInferenceRequest_) {
        pendingInferenceRequest_ = false;
        requestNextVideoFrame();
    }
}

void MainWindow::requestNextVideoFrame()
{
    if (!isPlaybackRunning_ || !controller_->hasOpenVideo()) {
        return;
    }

    if (inferenceInFlight_) {
        pendingInferenceRequest_ = true;
        return;
    }

    inferenceInFlight_ = true;
    QMetaObject::invokeMethod(videoWorker_,
                              "processFrame",
                              Qt::QueuedConnection,
                              Q_ARG(int, playbackGeneration_));
}

void MainWindow::displayImage(const QImage &image)
{
    ui->imageLabel->setPixmap(QPixmap::fromImage(image));
    ui->imageLabel->adjustSize();
}

void MainWindow::setPlaybackRunning(bool isRunning)
{
    isPlaybackRunning_ = isRunning;
    if (isPlaybackRunning_) {
        playbackTimer_.start();
        ui->playPauseButton->setText(QStringLiteral("Pause"));
    } else {
        playbackTimer_.stop();
        pendingInferenceRequest_ = false;
        ui->playPauseButton->setText(QStringLiteral("Play"));
    }
}
