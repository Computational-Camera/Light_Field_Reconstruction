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


#ifndef _MISC
#define _MISC

#include <opencv2/opencv.hpp>

bool devig(float* data_ptr1, float* data_ptr2, int size){
    for (int i=0; i<size; i++){
        //cout<<data_ptr1[i]<<" "<<data_ptr2[i]<<endl;
        data_ptr1[i] = data_ptr1[i]*data_ptr2[i];
        //cout<<data_ptr1[i]<<" "<<data_ptr2[i]<<endl;
    }
    return true;
}

bool gamma(unsigned short* data_ptr1, float* data_ptr2, float max, float gamma, int size){
    for (int i=0; i<size; i++){
        float value = data_ptr2[i]/max;
        if (value<1)
            value = 4095* pow (value, gamma); //should use lookup table to accelerate
        else 
            value = 4095;
        
        if (value<=4095)
            data_ptr1[i] = (unsigned short) (value+0.5);
        else
            data_ptr1[i] = 4095;             
    }
    return true;
}

bool gen_vig_mat(float* data_ptr, int size, float value){
    for (int i=0; i<size; i++)
        data_ptr[i] = value/data_ptr[i];
    return true;
}

bool clamp_normalize_mat( float* data_ptr, int size, float max){
    for (int i=0; i<size; i++)
            data_ptr[i] = data_ptr[i]/max;
    return true;
}


float percentage2value (float* data_ptr, int size, float percent)
{
    int number_bin=100;
	float* cdf_ptr=new float [number_bin]; 
    memset(cdf_ptr,0,(100)*sizeof(float));

    //cout<<size<<endl;
	float inv_size=1/float(size);
	float min_value= *min_element(data_ptr,data_ptr+size);
	float max_value= *max_element(data_ptr,data_ptr+size);	
	//cout<<min_value<<" "<<max_value<<endl;
	float bin_interval= (max_value - min_value) / (float (number_bin) - 1);// range of a bin

    float first_value = min_value - 0.5*bin_interval;// start of bin value
    //float last_value  = first_value + bin_interval * number_bin;// end of bin value
	//cout<<"max and min "<<max_value<<" "<<min_value<<endl; 
	for(int i=0;i<size;i++)
	{
		int index=int((*data_ptr++ - first_value)/bin_interval);  
		(*(cdf_ptr+index))++;
	}

	*(cdf_ptr)=*(cdf_ptr)*inv_size;
    float* temp_ptr=cdf_ptr+1;	
	for(int i=1;i<number_bin;i++)
	{
		*(temp_ptr)=(*(temp_ptr)*inv_size+*(temp_ptr-1));
		temp_ptr++;    
	}
	int index=0;
    while (*(cdf_ptr+index)<percent)
    {index=index+1;}

	float value=index*((max_value - min_value) / (float (number_bin) - 1))+first_value;
	//cout<<"value "<<value<<endl; 
	delete[] cdf_ptr;
	
    return value;
}


#endif
