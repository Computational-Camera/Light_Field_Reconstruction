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


#ifndef _CENTROID
#define _CENTROID

#include <opencv2/opencv.hpp>
#include "config.h"

using namespace std;
using namespace cv;

//just pick the maximum value
void find_peak(Mat img, Point& pos, int R){

    unsigned short temp_value=0;
    Point2f pos_temp;

    for (int j=-R;j<(R+1);j++){
        for(int i=-R;i<(R+1);i++){

            int idx = pos.x+i;
            int idy = pos.y+j;

            unsigned short pixel_value= img.at<unsigned short>(idy, idx);
            //cout<<pixel_value<<endl;
            if (pixel_value>temp_value){
                pos_temp = Point (idx, idy);
                temp_value=pixel_value;
                //cout<<pos_temp<<endl;
            }
        }
    }

    pos=pos_temp;
}

void centroid_refine(Mat img, GRID* grid){
    
    int R =3;
    for (size_t i=0; i<grid->pt_dst1.size(); i++){
        Point2f pt = grid->pt_dst1[i];
        
        int idx = pt.x;
        int idy = pt.y;

        float sx=0;
        float sy=0;
        float sxx =0;
        float syy =0;    
        float ox, oy;
        float s = 0;
        
        for (int j=-R;j<(R+1);j++){
            for(int i=-R;i<(R+1);i++){
                float temp = img.at<unsigned short>(idy+j, idx+i);
                temp = temp*temp;
                s  = s  + temp;
                sx = sx + i*temp;         
                sy = sy + j*temp;
                if (fabs(i)>0) sxx= sxx + temp;           
                if (fabs(j)>0) syy= syy + temp; 
                }
        }

        ox  = sx / sxx;            
        oy  = sy / syy;      

        grid->pt_dst1[i] = pt + Point2f(ox,oy);
    }
}


void build_grid(Mat img, PARA* para, GRID* grid){

    //erosion
    int erosion_size = 1;
    Mat dst;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                  cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                  cv::Point(erosion_size, erosion_size));
    erode(img,dst,element);

    Mat img_conv;
    int kernel_radius = 1;
    int kernel_size = 2*kernel_radius + 1;
    Mat kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
    filter2D(dst, img_conv, -1 , kernel, Point( -kernel_radius , -kernel_radius  ), 0, BORDER_DEFAULT );

    find_peak(dst, para->SP, para->R+2); // find the starting point


    float h_s = 2*para->R+1;
    float v_s = 0.866*h_s;//sqrt(3)/2
    float hh = 1;
    float vv = 1;
    Point pos_temp=para->SP;
    
    for(int j=0; j< para->ML_H;j++){
        for(int i=0; i< para->ML_W; i++){
            
            find_peak(img_conv, pos_temp, 2);                                
            grid->pt_dst1.push_back(Point2f(pos_temp));
            Point2f pt = Point2f(i*hh, j*vv);
            if (j%2==1) pt = pt + Point2f(hh/2, 0);
                grid->pt_src.push_back(pt);    
                pos_temp.x = pos_temp.x + h_s;
            }
                
            pos_temp.y=grid->pt_dst1[j* para->ML_W].y +v_s;
            pos_temp.x=grid->pt_dst1[j* para->ML_W].x;
            if(j%2==0) pos_temp.x = pos_temp.x+h_s/2;
            else       pos_temp.x = pos_temp.x-h_s/2;
        }
        
    centroid_refine(img_conv, grid);        
}

#endif

