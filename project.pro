
QT       += core
QT       -= gui

QT_CONFIG -= no-pkg-config
CONFIG    += link_pkgconfig

PKGCONFIG += opencv

TARGET     = ./bin/centre
TEMPLATE   = app

CONFIG    += console
CONFIG    += optimize_full
CONFIG    -= app_bundle

TEMPLATE = app

HEADERS		= ./src/*.h		  
SOURCES		= ./src/*.cpp
QMAKE_CXXFLAGS += -O3 -march=native -std=c++11 -m64 -pipe -ffast-math -Waggressive-loop-optimizations -Wall -fpermissive -fopenmp 


INCLUDEPATH +=  /usr/local/include/eigen3/
LIBS += -fopenmp -lz -lhdf5_serial -lX11
OBJECTS_DIR = ./obj
