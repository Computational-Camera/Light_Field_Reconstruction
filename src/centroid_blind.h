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



#ifndef _CENTROID_BLIND
#define _CENTROID_BLIND
#include <time.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "config.h"

using namespace std;
using namespace cv;

bool fft_img(Mat I, Mat& img_fft){

		Mat planes[] = {I, Mat::zeros(I.size(), CV_32F)};

		Mat complexI;
		merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

		dft(complexI, complexI);            // this way the result may fit in the source matrix
		
		split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
		magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
		Mat magI = planes[0];
        magI += Scalar::all(1);                    // switch to logarithmic scale
		log(magI, magI); 
		magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

		// rearrange the quadrants of Fourier image  so that the origin is at the image center
		int cx = magI.cols/2;
		int cy = magI.rows/2;

		Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
		Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
		Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
		Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

		Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
		q2.copyTo(q1);
		tmp.copyTo(q2);
		img_fft = magI.clone();

		return true;
}

/**
 *  ml_size_est (Mat img, PARA* para)
 *  Estimate the radius of the microlens by finding the harmonic peak of the spectrum.
 *
 */
float ml_size_est (Mat img, PARA* para) {

    img.convertTo(img, CV_32F);
    Mat img_fft;//amplitude
    fft_img(img, img_fft);
    //cout<<"img_fft size is "<<img_fft.size()<<endl;
    float sum =0;     
    Point2f pt_0 = Point2f(para->RAW_W/2, para->RAW_H/2);
    float radius_est = 350;
    float rrange     = 150;
    Point2f pt[6] = {Point2f(0, radius_est),                        Point2f( radius_est*sqrt(3)/2, radius_est/2), 
                     Point2f( radius_est*sqrt(3)/2, -radius_est/2), Point2f(0, -radius_est),
                     Point2f(-radius_est*sqrt(3)/2, -radius_est/2), Point2f(-radius_est*sqrt(3)/2, radius_est/2)};
    
    for (int k=0; k<6; k++){  
        pt[k] = pt[k] + Point2f(para->RAW_W/2, para->RAW_H/2);    
        double min, max; 
        Point min_loc, max_loc;
        minMaxLoc(img_fft(Rect(pt[k].x-rrange, pt[k].y-rrange, 2*rrange, 2*rrange)), &min, &max, &min_loc, &max_loc);	
        Point2f pc = Point2f(max_loc.x, max_loc.y) + Point2f(pt[k].x-rrange, pt[k].y-rrange); 
        float dist = para->RAW_W/sqrt((pc.x-pt_0.x)*(pc.x-pt_0.x)+(pc.y-pt_0.y)*(pc.y-pt_0.y));
        sum = sum + dist;  
        //cout<<"dist is "<< dist <<endl;             
    }  
    return sum/6.0f;
}

/**
 *  gen_score_map()
 *  Generate the score map based on the local estimator
    vector<Mat>& img   a vector containing 4 by 4 tilt images
    vector<Mat>& score a vector containing 4 by 4 tilt score maps
 *
 */

inline bool gen_score_map(Mat img, Mat& score, float D){
    //magnification is 4
    double idy[6]={4*D*sqrt(3)*1.0f/3.0f,-4*D*sqrt(3)*1.0f/3.0f, 4*D*sqrt(3)*1.0f/6.0f,4*D*sqrt(3)*1.0f/6.0f,-4*D*sqrt(3)*1.0f/6.0f, -4*D*sqrt(3)*1.0f/6.0f}; 
    double idx[6]={                    0,                     0,               0.5*4*D,             -0.5*4*D,               0.5*4*D,               -0.5*4*D};

    score= Mat(img.size(), CV_32FC1);
    const char border = 32;
    #pragma omp parallel for
    for(int j=border;j<img.rows-border;++j){
	    for(int i=border;i<img.cols-border;++i){

		    double avg=0;
		    double v;
		    double vmax = 0;
		    double vmin = numeric_limits<double>::max();
		    for (int t=0;t<6; ++t){
                v=(double)img.at<float>(round(j+idy[t]),round(i+idx[t]));
                avg=avg+v;
                vmax = (v>vmax) ? v :vmax;
                vmin = (v<vmin) ? v :vmin;				       
		    }
		    score.at<float>(j,i)=(avg-vmax-vmin)/4;//remove two outliers vmax and vmin		    
	    }
    } 
    return true;
}


/**
 *  Generate the ideal grid vector
 *  vector<Point2d> pos vector of microlens centres in ideal grid
 *  Rect ROI
 *  PARA* para configuration paramters
 *  float magnification grid 
 *  int inc 
 */
bool gen_pos(vector<Point2d>& pos, Rect ROI, float magnification, int inc){ //no maginification

    float hh = magnification;
    float vv = magnification;
    //cout<<   ROI.y <<" "<<ROI.height<<" "<<ROI.x<<" "<<ROI.width <<endl;
	for(int j=ROI.y; j<ROI.y+ROI.height; j=j+inc){
        for(int i=ROI.x; i<ROI.x+ROI.width; i=i+inc){
                                         
            Point2f pt = Point2f(i*hh, j*vv);
            if (j%2==1) pt = pt + Point2f(hh/2, 0);
                pos.push_back(pt);                            
        }          
    }	
	return true;
}

/**
 *  Generate the real grid vector
 *  vector<Point2d> pos vector of microlens centres in ideal grid
 *  vector<Point2d> pos2 vector of microlens centres in real grid
 *  double* trans_vec translation matrix (2x3)
 *  para configuration paramters
 */
bool gen_pos2(vector<Point2d> pos, 
              vector<Point2d>& pos2, 
              double* tran_vec, 
              double theta, 
              double scale,
              PARA* para){ //no maginification

    pos2.resize(pos.size());
    
    for(size_t k=0; k<pos.size(); k++){

        pos2[k].x= -para->RAW_W*4/2 + (pos[k].x*tran_vec[0]+tran_vec[2]);
		pos2[k].y=  para->RAW_H*4/2 - (pos[k].y*tran_vec[4]+tran_vec[5]);							
							
		pos2[k].x = pos2[k].x * cos(theta) - pos2[k].y * sin(theta); 	
		pos2[k].y = pos2[k].x * sin(theta) + pos2[k].y * cos(theta); 	
							
		pos2[k].x=  para->RAW_W*4/2 + scale*pos2[k].x ;
		pos2[k].y=  para->RAW_H*4/2 - scale*pos2[k].y ;	
                            
    }
	return true;
}

bool save_para(double* src, double* dst){
    for (int i=0;i<6;i++)
	    dst[i]=src[i];
    return true;
}

/**
 *  brute_force_search_2d()
 *  Brue Force Search in 2D
 *  return 2 updated parameters
 */
bool brute_force_search_2d (int* itn,
							double* step,
							double* tran_vec,
							vector<Point2d> pos,
							const Mat& img){
    double tran_vec_temp[6];
    double tran_vec_optimum[6]={0,0,0,0,0,0};
    save_para(tran_vec,tran_vec_temp);

    Mat cost_mat = Mat::zeros(2*itn[2]+1,2*itn[5]+1,CV_32F);
    
    #pragma omp parallel for   
    for(int g=-itn[2];g<(itn[2]+1);++g){
	    for(int h=-itn[5];h<(itn[5]+1);++h){
	
		    tran_vec_temp[2]=tran_vec[2]+g*step[2];
		    tran_vec_temp[5]=tran_vec[5]+h*step[5];
            //cout<<tran_vec[2]<<" "<<tran_vec[5]<<" "<<g<<" "<<h<<" "<<tran_vec_temp[2]<<" "<<tran_vec_temp[5]<<endl;
		    double next=0; 
		    vector<Point2d> pos_2d;
		    pos_2d.resize(pos.size());

		    for(size_t kk=0;kk<pos.size();kk++){	    		
                pos_2d[kk].x= pos[kk].x*tran_vec_temp[0]+pos[kk].y*tran_vec_temp[1]+tran_vec_temp[2];
                pos_2d[kk].y= pos[kk].x*tran_vec_temp[3]+pos[kk].y*tran_vec_temp[4]+tran_vec_temp[5];
                next=next+img.at<float>(round(pos_2d[kk].y), round(pos_2d[kk].x));
		    }
            cost_mat.at<float>(g+itn[2],h+itn[5])=next;		
            vector<Point2d>().swap(pos_2d);
	    }
    }
    
    double min, max; 
    Point min_loc, max_loc;
    minMaxLoc(cost_mat, &min, &max, &min_loc, &max_loc);	   
    tran_vec_temp[2]=tran_vec[2]+(min_loc.y-itn[2])*step[2];
	tran_vec_temp[5]=tran_vec[5]+(min_loc.x-itn[5])*step[5];
    //mat2hdf5 ( "./debug/2d_score.h5", "data", H5T_NATIVE_FLOAT, float(), score_map);
    save_para(tran_vec_optimum,tran_vec_temp);
    return true;
}

/**
 *  brute_force_search_theta()
 *  Brue Force Search in 4D
 *  return 4 update parameters
 */

bool brute_force_search_theta (float& theta,
                               float& scale,
                               PARA* para,
							   double* tran_vec,
							   vector<Point2d> pos,
						       vector<Point2d>& pos2,	
							   Mat img){

    Mat cost_map = Mat::zeros(2*100+1,2*100+1,CV_32F);

    int WW = para->RAW_W;
    #pragma omp parallel for   	
	for(int m=-100;m<=100;m++){
        for(int n=-100;n<=100;n++){

		double next=0; 
		vector<Point2d> pos_2d;
		pos_2d.resize(pos.size());
		float thetan=n/10000.0f;
		float scalen=1+m/10000.0f;
		float cost = scalen*cos(thetan);
		float sint = scalen*sin(thetan);
		
		for(size_t kk=0;kk<pos.size();kk++){	    
		
		  	pos_2d[kk].x= -WW*4/2 + (pos[kk].x*tran_vec[0]+tran_vec[2]);
			pos_2d[kk].y=  WW*4/2 - (pos[kk].y*tran_vec[4]+tran_vec[5]);							
							
			pos_2d[kk].x = pos_2d[kk].x * cost - pos_2d[kk].y * sint; 	
			pos_2d[kk].y = pos_2d[kk].x * sint + pos_2d[kk].y * cost; 	
							
		  	pos_2d[kk].x=  WW*4/2 + pos_2d[kk].x ;
			pos_2d[kk].y=  WW*4/2 - pos_2d[kk].y ;	
																					
			Point p=Point(round(pos_2d[kk].x),round(pos_2d[kk].y));
			next=next+img.at<float>(p);									
		}
		cost_map.at<float>(m+100,n+100)=next;
		vector<Point2d>().swap(pos_2d);
	    }
    }
    
    double min, max; 
    Point min_loc, max_loc;
    minMaxLoc(cost_map, &min, &max, &min_loc, &max_loc);	       
 
    theta = (min_loc.x-100)  /10000.0f;
	scale = 1+(min_loc.y-100)/10000.0f;
	float cost = scale*cos(theta);
	float sint = scale*sin(theta);			   
    pos2.resize(pos.size());
  	for(size_t kk=0;kk<pos.size();kk++){	    
									
		  	pos2[kk].x= -WW*4/2 + (pos[kk].x*tran_vec[0]+tran_vec[2]);
			pos2[kk].y=  WW*4/2 - (pos[kk].y*tran_vec[4]+tran_vec[5]);							
							
			pos2[kk].x = pos2[kk].x * cost - pos2[kk].y * sint; 	
			pos2[kk].y = pos2[kk].x * sint + pos2[kk].y * cost; 	
							
		  	pos2[kk].x=  WW*4/2 + pos2[kk].x ;
			pos2[kk].y=  WW*4/2 - pos2[kk].y ;	
																					
			//Point p=Point(round(pos2[kk].x),round(pos2[kk].y));	
	}  
    																		
    //mat2hdf5 ( "./debug/2d_score2.h5", "data", H5T_NATIVE_FLOAT, float(), cost_map);
    return true;
}

/**
 *  refine the position individually
 *  from the global fitting
 */

bool refine_pos(vector<Point2d>& pos, const Mat& img){

	for(size_t k=0; k<pos.size(); k++){
	    
	    float temp=10000;
	    Point2d pt_temp;
	    for (int j=-8; j<9; j++)
	        for (int i=-8; i<9; i++){
	        	//cout<<pos[k]<<endl;
	            float value = img.at<float>(pos[k]+Point2d(i,j));

	            if (temp>value){
	                temp =value;
	                pt_temp = Point2d(i,j);
	            }
	                
	        }

	    pos[k]= pos[k] + pt_temp;	        		
	}

    return true;
}

/**
 * grid_optimization()
 * Find the opimized paramters by a complete search
 * return a six paramters
 * img is the cost map
 * img2 is the content image
*/
bool grid_optimization(Mat& img,  Mat& H,
                       vector<Point2d>& pos_vec,  vector<Point2d>& pos_vec2,
                       double* tran_mat_vec, PARA* para){

    //initial condition
    int    itn [6]    ={  100,   100,  200,   100,   100, 200};
    double step[6]    ={0.002, 0.002,  0.1, 0.002, 0.002, 0.1};

    cout<<"Optimization Step 1..."<<endl; //translation estimation
    gen_pos(pos_vec, Rect(para->ML_W*3/8,para->ML_H*3/8,para->ML_W/4,para->ML_H/4), 4, 2);//centeral region      
    tran_mat_vec[2]=4.0f*tran_mat_vec[2]; //compensation for magnification
    tran_mat_vec[5]=4.0f*tran_mat_vec[5];

    brute_force_search_2d(itn,step,tran_mat_vec,pos_vec,img);
    vector<Point2d>().swap(pos_vec);

    cout<<"Optimization Step 2..."<<endl;//coarse globally estimation 
    gen_pos(pos_vec, Rect(0,0,para->ML_W,para->ML_H), 4, 4);//centeral region   
    float theta=0;
    float scale=0;
    brute_force_search_theta(theta, scale, para, tran_mat_vec, pos_vec, pos_vec2, img);
    cout<<"theta is "<<theta<<" "<<scale<<endl;
    vector<Point2d>().swap(pos_vec);
    vector<Point2d>().swap(pos_vec2);

    gen_pos(pos_vec, Rect(0,0,para->ML_W,para->ML_H), 4, 1);
    gen_pos2(pos_vec, pos_vec2, tran_mat_vec, theta, scale, para);
    refine_pos(pos_vec2, img);

    cout<<"Optimization Step 3..."<<endl;
    vector<Point2d> pos_vec3;
    pos_vec3.resize(pos_vec.size());

    for(size_t kk=0;kk<pos_vec3.size();kk++){	    		
        pos_vec3[kk].x = pos_vec2[kk].x/4;
        pos_vec3[kk].y = pos_vec2[kk].y/4;
        pos_vec[kk].x  = pos_vec[kk].x/4;
        pos_vec[kk].y  = pos_vec[kk].y/4;
    }		

    H = findHomography( pos_vec, pos_vec3, RANSAC);  

    return true;
}
 

void build_grid_blind(Mat img, PARA* para, GRID* grid){

    vector<Point2d> pos_vec, pos_vec2;

    float d = ml_size_est (img, para);
    //cout << "d is " <<d<<" "<< d *2 /sqrt(3)<<endl;

    vector<Mat> img_tilt;//(IMG_H*MAG,IMG_W*MAG,CV_32FC1);
    Mat img_blur;
    GaussianBlur(img, img_blur, Size(5, 5), 0, 0);
    Mat img4;
    resize(img_blur, img4, Size(), 4, 4, INTER_CUBIC);
    Mat score4(img4.size(), CV_32F);

    timespec ts[5];
    clock_gettime(CLOCK_REALTIME, &ts[0]);
    //upsample by 4x4 titls ignore  extra boundary
    #pragma omp parallel for   
    for (int j=0; j<4; j++)
        for (int i=0; i<4; i++){
            Rect ROI  = Rect(i* para->RAW_W, j*para->RAW_H, para->RAW_W, para->RAW_H);
            Mat img44;
            img4(ROI).convertTo(img44, CV_32F);
            Mat cost_map;   
            gen_score_map(img44, cost_map, d *2 /sqrt(3));
            cost_map.copyTo(score4(Rect(ROI)));
        }
    
    double tran_mat_vec[6]={d *2 /sqrt(3),0,40,0,d,40};

    Mat H;
    grid_optimization(score4, H, pos_vec, pos_vec2, tran_mat_vec, para);

    vector<Point2d>().swap(pos_vec);    
    gen_pos(pos_vec, Rect(0,0,para->ML_W,para->ML_H), 1, 1);
    Mat pp;

    grid->pt_src.resize(pos_vec.size());     
    for (size_t i=0; i<pos_vec.size(); i++)
        grid->pt_src[i] = (Point2f) pos_vec[i];    

    perspectiveTransform(Mat(grid->pt_src), pp, H);

    vector<Point2f>().swap(grid->pt_src);    
    vector<Point2f>().swap(grid->pt_dst2);
    grid->pt_dst2.resize(pos_vec.size());  
    for (size_t i=0; i<pos_vec.size(); i++){
        grid->pt_dst2[i] = pp.at<Point2f>(0,i); 
    }

    clock_gettime(CLOCK_REALTIME, &ts[1]);   
    float diff_time = (ts[1].tv_nsec - ts[0].tv_nsec);
	diff_time = (diff_time>0) ? diff_time/1000000000.f   + (ts[1].tv_sec - ts[0].tv_sec): 
                                diff_time/1000000000.f+1     + (ts[1].tv_sec - ts[0].tv_sec)-1;
	//cout << " Time spent " << diff_time<<"s"<<endl;     
}

#endif
