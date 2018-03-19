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


#ifndef _RECONSTRUCTION
#define _RECONSTRUCTION

#include <opencv2/opencv.hpp>
#include "config.h"
#include "misc.h"
using namespace std;
using namespace cv;

unsigned short bilinear(unsigned short* data, float index_y, float index_x, int W){

    float value;
    float cof[4];
    float a,b,c,d;

    int y = int(index_y);
    int x = int(index_x);
    int base = y*W+x;

    a = data[base];
    b = data[base+1];
    c = data[base+W];
    d = data[base+W+1];

    cof[0]=(1-(index_y-y))*(1-(index_x-x));
    cof[1]=(1-(index_y-y))*(index_x-x);
    cof[2]=    (index_y-y)*(1-(index_x-x));
    cof[3]=    (index_y-y)*(index_x-x);

    value= (cof[0]*a+cof[1]*b+cof[2]*c+cof[3]*d);
    return (unsigned short) value;
}

bool grad_difference (unsigned short* img,  size_t j, size_t i, 
                      int WHST, size_t IMG_W, size_t T,size_t S, int m, int n){
             
    int ii = ((j%2)==0) ? (i-1)/2 : i/2; 
    int c  = ((j%2)==0) ?  2      :  0; //ugly implementation

    float grad_x_max=0;
    float grad_y_max=0;
    
    for (int k=0; k<3; k++){
        float grad_x = abs( img[k*WHST+(j*T+m)*IMG_W*S+ii*S+n]-
                                      img[k*WHST+(j*T+m)*IMG_W*S+(ii-1+c)*S+n]); 
        if (grad_x>grad_x_max)   
            grad_x_max = grad_x;                                 
        float grad_y = abs( img[k*WHST+((j-1)*T+m)*IMG_W*S+(ii)*S+n]-
                                      img[k*WHST+((j+1)*T+m)*IMG_W*S+(ii)*S+n]);
        if (grad_y>grad_y_max)   
            grad_y_max = grad_y;    
    }
    bool value =  ((4*grad_y_max < grad_x_max)?true:false);
    return value; 
}

void  devig_simple(Mat raw, Mat raw_ref, Mat& rawc){

    unsigned short *rawc_ptr    = (unsigned short*) rawc.data;      

    Mat img, img2;
    raw_ref.convertTo(img, CV_32F);
    raw.convertTo(img2, CV_32F);
    Scalar mean,stddev; 
    
    //float min= int(percentage2value ((float*) img.data,  img.total(), 0.001));//<<endl;
    //float max= int(percentage2value ((float*) img.data,  img.total(), 0.999));//<<endl;  
    clamp_normalize_mat( (float*) img.data, img.total(), 4095.0);
    meanStdDev(img,mean,stddev,cv::Mat());
    //float mean50= percentage2value ((float*) img.data,  img.total(), 0.50);//<<endl;       
    
    gen_vig_mat((float*) img.data, img.total(), mean.val[0]);
    devig((float*) img2.data, (float*) img.data, img.total());
    double min, max;
    minMaxLoc(img2, &min, &max); 
    gamma(rawc_ptr, (float*) img2.data, 4095, 0.50, img2.total());
    
  }


void decode(Mat img_colour, Mat& multiview, GRID grid, PARA para){

    size_t IMG_W = para.ML_W*2;  //final image is 2 times larger due to hexgonal interpolation
    size_t IMG_H = para.ML_H;
    size_t ML_W  = para.ML_W; 
    size_t ML_H  = para.ML_H;
    size_t S     = para.S;
    size_t T     = para.T;

    int WHST  = IMG_W*IMG_H*S*T;
    
    unsigned short*  lf_buf        = new unsigned short[3*WHST];
    unsigned short *multiview_buf  = new unsigned short[3*WHST];
    
    multiview = Mat::zeros(IMG_H*T, IMG_W*S, CV_16UC3);
    Mat bgr[3];   //destination array
    split(img_colour,bgr);//split source  

    //4D array registration
    #pragma omp parallel for      
    for (size_t j=0; j<ML_H; j++){
        for (size_t i=0; i<ML_W; i++){
            float index_y,index_x;

            if (grid.pt_dst2.size()>0){
                 index_y= grid.pt_dst2[j*ML_W+i].y; //_dst2
                 index_x= grid.pt_dst2[j*ML_W+i].x;
            }
            else{
                 index_y= grid.pt_dst1[j*ML_W+i].y; //pt2
                 index_x= grid.pt_dst1[j*ML_W+i].x; 
            }      
            
            int ix = int (index_x);
            int iy = int (index_y);
            
            Mat subimg;
            
            for (int k=0; k<3; k++){
                resize(bgr[k](Rect(ix-5,iy-5,11,11)),subimg,Size(),4,4, INTER_CUBIC);
                //cout<<subimg<<endl;
                int idx_new = 22+int(4*(index_x-ix));
                int idy_new = 22+int(4*(index_y-iy));
                
                for (int m=-4; m<5; m++)
                    for (int n=-4; n<5; n++){
                      int temp_index = (j*T+(m+4))*ML_W*S + i*S + (n+4); 
                      lf_buf[temp_index+ k*WHST/2] = subimg.at<unsigned short>(idy_new+4*m, idx_new+4*n);
                    }
            }          
    
         }
     }
  
    bool grad_h_v;
    //cout<<"=============================================="<<endl;
    for (size_t j=1; j<(IMG_H-1); j++){
        for (size_t i=1; i<(IMG_W-1); i++){
            if ((j%2)==0){  // o x o x o x
                if ((i%2)==1){  // interpolate
                       
                     for (size_t m=0; m<T; m++){
                        for (size_t n=0; n<S; n++){                            
                             grad_h_v= grad_difference (lf_buf, j, i, WHST/2, ML_W, T, S, m, n);  
                                
                             for (size_t k=0; k<3; k++){                                 
                                multiview_buf[k*WHST+(j*T+m)*IMG_W*S+i*S+n] = 
                                grad_h_v?
                                 (lf_buf[k*WHST/2+((j-1)*T+m)*ML_W*S+(i-1)/2*S+n]
                                 +lf_buf[k*WHST/2+((j+1)*T+m)*ML_W*S+(i-1)/2*S+n])/2:
                                ( lf_buf[k*WHST/2+(j*T+m)*ML_W*S+(i-1)/2*S+n]
                                 +lf_buf[k*WHST/2+(j*T+m)*ML_W*S+(i+1)/2*S+n])/2;                     
                            }
                        }
                
                        
                    } 
                }                   
                else{
                    for (size_t m=0; m<T; m++){
                        for (size_t n=0; n<S; n++){
                            for (size_t k=0; k<3; k++){
                                multiview_buf[k*WHST+(j*T+m)*IMG_W*S+i*S+n] = 
                                lf_buf[k*WHST/2+(j*T+m)*ML_W*S+(i/2)*S+n];
                            }
                        }
                    }
                }                                        
            }   
            else{      // x o x o x o
                if((i%2)==0){   //interpolate
                    for (size_t m=0; m<T; m++){
                        for (size_t n=0; n<S; n++){
                            grad_h_v= grad_difference (lf_buf,j,i,WHST/2,ML_W,T,S,m,n);
                            for(size_t k=0; k<3; k++){
                                multiview_buf[k*WHST+(j*T+m)*IMG_W*S+i*S+n] = 
                                grad_h_v?
                                (lf_buf[k*WHST/2+((j-1)*T+m)*ML_W*S+i/2*S+n]
                                +lf_buf[k*WHST/2+((j+1)*T+m)*ML_W*S+i/2*S+n])/2:
                                (lf_buf[k*WHST/2+(j*T+m)*ML_W*S+(i/2-1)*S+n]
                                +lf_buf[k*WHST/2+(j*T+m)*ML_W*S+i/2*S+n])/2;                              
                            }
                        }
                    }

                }
                else{   //copy
                    for (size_t m=0; m<T; m++){
                        for (size_t n=0; n<S; n++){
                            for(size_t k=0; k<3; k++){
                                multiview_buf[k*WHST+(j*T+m)*IMG_W*S+i*S+n] =
                                lf_buf[k*WHST/2+(j*T+m)*ML_W*S+(i-1)/2*S+n];
                            }
                        }
                    }
                }
           }
        }
    }      
         
    //ml array->multiview    
    for (size_t j=0; j<IMG_H; j++){
        for (size_t i=0; i<IMG_W; i++){
            for (size_t m=0; m<T; m++){
                for (size_t n=0; n<S; n++){
                    
                    int index_2=(j*T+m)*IMG_W*S + i*S+n;                    
                    multiview.at<Vec3w>(m*IMG_H+j, n*IMG_W +i) =
                    Vec3w(multiview_buf[index_2],multiview_buf[WHST+index_2],multiview_buf[2*WHST+index_2]);
                    //if ((multiview_buf[2*WHST+index_2]==212)&&(multiview_buf[WHST+index_2]==67))
                    //    cout<<j<<" "<<i<<" "<<m<<" "<<n<<endl;
                }
            }
        }
    } 
}
    
#endif
