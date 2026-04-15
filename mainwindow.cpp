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
    statusBar()->showMessage(QStringLiteral("Ready. Open an image to run detection."));
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

void MainWindow::displayImage(const QImage &image)
{
    ui->imageLabel->setPixmap(QPixmap::fromImage(image));
    ui->imageLabel->adjustSize();
}
