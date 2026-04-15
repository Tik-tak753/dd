QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    controller/appcontroller.cpp \
    detection/stubdetector.cpp \
    main.cpp \
    mainwindow.cpp \
    utils/cvqtutils.cpp

HEADERS += \
    controller/appcontroller.h \
    detection/detection.h \
    detection/idetector.h \
    detection/stubdetector.h \
    mainwindow.h \
    utils/cvqtutils.h

FORMS += \
    mainwindow.ui

# OpenCV
OPENCV_DIR = C:/cv/opencv/build

INCLUDEPATH += $$OPENCV_DIR/include

win32-msvc {
    CONFIG(debug, debug|release) {
        LIBS += -L$$OPENCV_DIR/x64/vc16/lib \
                -lopencv_world4100d
    } else {
        LIBS += -L$$OPENCV_DIR/x64/vc16/lib \
                -lopencv_world4100
    }
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
