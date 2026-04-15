#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "controller/appcontroller.h"

#include <QFileDialog>
#include <QImage>
#include <QPixmap>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , controller_(std::make_unique<AppController>())
{
    ui->setupUi(this);
    setWindowTitle(QStringLiteral("Drone Detection Demo"));

    connect(ui->openImageButton, &QPushButton::clicked, this, &MainWindow::onOpenImageClicked);
    connect(ui->loadModelButton, &QPushButton::clicked, this, &MainWindow::onLoadModelClicked);
    statusBar()->showMessage(QStringLiteral("Ready. %1").arg(controller_->currentDetectorStatus()));
}

MainWindow::~MainWindow()
{
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

void MainWindow::displayImage(const QImage &image)
{
    ui->imageLabel->setPixmap(QPixmap::fromImage(image));
    ui->imageLabel->adjustSize();
}
