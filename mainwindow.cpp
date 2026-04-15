#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <opencv2/core.hpp>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    cv::Mat img(200, 300, CV_8UC3);
    setWindowTitle("Qt + OpenCV OK");
}

MainWindow::~MainWindow()
{
    delete ui;
}