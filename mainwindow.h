#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

#include <memory>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class AppController;

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
    void onPlaybackTick();

private:
    void displayImage(const QImage &image);
    void setPlaybackRunning(bool isRunning);

    Ui::MainWindow *ui;
    std::unique_ptr<AppController> controller_;
    QTimer playbackTimer_;
    bool isPlaybackRunning_ = false;
};

#endif // MAINWINDOW_H
