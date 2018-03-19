/*
    Multi-view Image Restoration
 
	"Multi-view Image Restoration From Plenoptic Raw Images"
	 Shan Xu, Zhi-Liang Zhou and Nicholas Devaney
	In Asian Conference on Computer Vision 2014
    Emerging Topics In Image Restoration and Enhancement workshop    
    Author: Shan Xu <xushan2011@gmail.com>
    Copyright (C) 2014-2018 
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */



#ifndef _CONFIG
#define _CONFIG

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef struct { 
    int RAW_W; //width  of raw image
    int RAW_H; //height of raw image
    int ML_W;  //number of vertical microlenses
    int ML_H;  //number of horizontal microlenses
    int S;     //number of vertical views
    int T;     //number of horizontal views    
    int R;     //approximate radius of the microlens in pixel
    int BLACK;
    Point SP;  // Postion to start the centre detection    
}PARA;

typedef struct { 
    vector<Point2f> pt_src; //centres coordinates
    vector<Point2f> pt_dst1; //measured centres       
    vector<Point2f> pt_dst2; //measured centres     
    vector<Point2f> pt;//global grid centres  
    vector<Point2f> pt2; //line centres                
    Mat H;//homography matrix    
}GRID;


/**
    Load the configuration file.
    @filename        file name to load. Should be a xml file.
*/

bool config_read(string filename, PARA* para){
    
    //===load configuration file from xml file
    FileStorage fs;
    fs.open(filename, FileStorage::READ);
    
    para->RAW_W=fs["RAW_W"];  
    para->RAW_H=fs["RAW_H"];     
    para->ML_W =fs["ML_W"]; 
    para->ML_H =fs["ML_H"]; 
    para->S    =fs["S"]; 
    para->T    =fs["T"];     
    para->R    =fs["R"];   
    para->SP   =Point(fs["SPX"],fs["SPY"]);       
    para->BLACK=fs["BLACK"];            
    fs.release();
    return 0;
}

#endif

