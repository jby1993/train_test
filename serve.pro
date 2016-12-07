QT += core
QT -= gui

CONFIG += c++11

TARGET = SythenticImagesWith3DMM
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += servemain.cpp \
    train_test.cpp \
    featuredetector.cpp \
    siftdectector.cpp \

HEADERS += \
    tri_mesh.h \
    train_test.h \
    featuredetector.h \
    siftdectector.h \

INCLUDEPATH +=./ \
             /usr/include \
            /home/jby/eigen3 \
            /usr/local/include \
            /home/jby/vlfeat-0.9.20/vl \


LIBS += -L/usr/lib \
        -L/usr/local/lib \
        -L/usr/local/lib/OpenMesh \
        -L/home/jby/vlfeat-0.9.20/bin/glnxa64 \

LIBS+=-lOpenMeshCore \
        -lOpenMeshTools \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \
        -lvl \
