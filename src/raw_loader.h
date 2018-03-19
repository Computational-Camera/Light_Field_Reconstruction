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


#ifndef _LF_RAW_LOADER
#define _LF_RAW_LOADER

#include <iostream>
#include <stdio.h>
#include <cstring>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define FILESIZE    18874368 //to determin pack or upack mode
// the format of Mat will be 16bit unsigned short with the range (0-4095)
void raw2buf(string raw_name, Mat& img){

    //load lytro RAW 16bit unpackedfile
    FILE *fp;

    fp=fopen(raw_name.c_str(),"r+b");
    if (fp!=NULL){

	fseek (fp, 0, SEEK_END); 
	long size=ftell (fp);
	fseek(fp, 0, SEEK_SET);//back to begining

	if (size>FILESIZE){ //not packed (16bit mode)
	    for(int j=0;j<img.rows;j++){
		    for(int i=0;i<img.cols;i++){
		
		        unsigned char temp1=getc(fp);
		        unsigned char temp2=getc(fp);
		        img.at<unsigned short>(j,i)= (((temp2<<8)+temp1)>>4);// lower 4bit is zeros
		    }
	    }
        cout<<raw_name<<" is loaded in 16 bit mode. "<<endl;
	}
	else{
	// 12bit pack mode
	    for(int j=0;j<img.rows;j++){
		    for(int i=0;i<img.cols/2;i++){
		        unsigned char temp1=getc(fp);
		        unsigned char temp2=getc(fp);
		        unsigned char temp3=getc(fp);
		        //normalize to 1
		        img.at<unsigned short>(j,2*i)   =(((0xf0&temp2)>>4)+(temp1<<4));//2^12;
		        img.at<unsigned short>(j,2*i+1) =(((0x0f&temp2)<<8)+temp3)     ;//2^12;
		    }
		}
        cout<<raw_name<<" is loaded in 12 bit pack mode. "<<endl;
	}
	fclose(fp);
    }
    else  cout<<" failed to load "<<raw_name<<endl;
}



void buf2raw(string raw_name, const Mat& img){
    
//store lytro RAW data as 12bit packed
    FILE *fp;

    fp=fopen(raw_name.c_str(),"w+b");
    
    for(int j=0;j<img.rows;j++){
        for(int i=0;i<img.cols;i++){
            unsigned short   d = (unsigned short)(img.at<float>(j,i)+0.5)<<4;
            unsigned char temp1=(unsigned char) ( (d&0x00ff)    );
            unsigned char temp2=(unsigned char) ( (d&0xff00)>>8 );

            putc(temp1,fp);
            putc(temp2,fp);          
	   }
    }

   fclose(fp);
}

#endif
