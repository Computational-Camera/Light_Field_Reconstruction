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


#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <cstring>

#include "config.h"
#include "h5_io.h"
#include "centroid.h"
#include "centroid_blind.h"
#include "reconstruction.h"
#include "raw_loader.h"
using namespace std;
using namespace cv;


/* 
usage:  
./bin/centre 0          ./data/office_ref.raw  ./data/office.raw ./configs/config1.xml ./output/office.jpg
           argv[1]             argv[2]            argv[3]                 argv[4]           argv[5]  

argv[1]  Reconstruction method  0: blind, 1 non-blind
argv[2]  reference raw image file 
argv[3]  content image file
argv[4]  configuration file
argv[5]  output multiview image

*/

int main(int argc, const char **argv){

    Mat ref_colour, ref_gray, ref_bayer;
    Mat img_colour, img_gray, img_bayer, imgc;

    PARA para;
    GRID grid;
    
    if (argc!=6){
        cout<<" Check Your Input Parameters "<<endl;
        return false;   
    }
    
    int option = atoi(argv[1]);

    cout<<"===========  Loading the Raw Data   ============="<<endl;
    config_read(argv[4], &para);
    ref_bayer=Mat(para.RAW_H,para.RAW_W,CV_16U);
    img_bayer=Mat(para.RAW_H,para.RAW_W,CV_16U);
        
    raw2buf(argv[2], ref_bayer);
    raw2buf(argv[3], img_bayer);
    cvtColor(img_bayer, img_colour, CV_BayerRG2BGR);

    cout<<"===========  Building the Grid   ================"<<endl;
    if (option==1){
        cvtColor(ref_bayer,  ref_colour, CV_BayerRG2BGR);
        cvtColor(ref_colour, ref_gray,  CV_BGR2GRAY);
        build_grid(ref_gray, &para, &grid);
        //build_grid_blind(ref_bayer, &para, &grid);  
    }
    else
    {
        build_grid_blind(img_bayer, &para, &grid);  
    }

    cout<<"============ Vignetting Modeling  ==============="<<endl;
    imgc  = Mat(para.RAW_H,para.RAW_W, CV_16U);
    devig_simple(img_bayer, ref_bayer, imgc);

    cout<<"========== Light Field Reconstruction  =========="<<endl;

    cvtColor(ref_bayer, ref_colour, CV_BayerRG2BGR);
    cvtColor(imgc,      img_colour, CV_BayerRG2BGR);      
    Mat multiview;
    decode(img_colour, multiview, grid, para);
    resize(multiview, multiview, Size(640*9,640*9), INTER_CUBIC);
    imwrite(argv[5], multiview*16);

    cout<<"================== Finish ======================="<<endl;
    return 0;
}




