#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

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
    void onLoadModelClicked();

private:
    void displayImage(const QImage &image);

    Ui::MainWindow *ui;
    std::unique_ptr<AppController> controller_;
};

#endif // MAINWINDOW_H
