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

#ifndef _HDF5
#define _HDF5

#include <hdf5/serial/hdf5.h>
#include <hdf5/serial/hdf5_hl.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
    Save Mat variable to HDF5 file.
    @file_name          File name to store
    @file_dataset_name  HDF5 Data set name
    @d_type             H5T_NATIVE_INT H5T_NATIVE_CHAR H5T_NATIVE_DOUBLE
    @TYPE               the type of Mat
    @img                OpenCV Mat 
*/
template<class TYPE>
void  mat2hdf5      ( const char* file_name, 
					  const char* file_dataset_name,
					 hid_t d_type,
					 TYPE,
                     Mat& img){

    hid_t       file, dataset,dataset_config; /* file and dataset handles */
    hid_t       datatype, dataspace;   /* handles */
    hsize_t     dimsf[2];              /* dataset dimensions */

    file = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    dimsf[0] = img.rows;
    dimsf[1] = img.cols;

    dataspace = H5Screate_simple(2, dimsf, NULL);

    datatype = H5Tcopy(d_type);
    H5Tset_order(datatype, H5T_ORDER_LE);
    dataset_config = H5Pcreate(H5P_DATASET_CREATE);
    dataset = H5Dcreate2(file, file_dataset_name, datatype, dataspace,
              H5P_DEFAULT, dataset_config, H5P_DEFAULT);

    TYPE* temp=new TYPE[img.rows*img.cols];
    for (int j = 0; j <img.rows; j++)
	    for (int i = 0; i < img.cols; i++)
            temp[j*img.cols+i]=img.at<TYPE>(j,i);

    H5Dwrite(dataset, d_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);

    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Fclose(file);
    delete[] temp;
}

/**
    Load HDF5 file to Mat variable.
    @file_name          File name to load
    @file_dataset_name  HDF5 Dataset name
    @data               OpenCV Mat need to be float
*/
void  hdf52mat (const char* file_name, 
                const char* file_dataset_name,
                Mat& data){


   hid_t       file_id, dataset_id; 
   file_id = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);
   dataset_id = H5Dopen(file_id, file_dataset_name,H5P_DEFAULT);

   // Read the dataset. 
   double* data_vec = new double[data.cols*data.rows];
   H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
                    data_vec);

        cout<<data.cols<<" "<<data.rows<<endl;
       for (int i=0; i<data.rows; i++){
           data.at<double>(i,0)= data_vec[2*i];
           data.at<double>(i,1)= data_vec[2*i+1]; 
           //cout<<data.at<double>(i,0)<<" "<<data.at<double>(i,1)<<" " <<endl;
	   }

       
   H5Dclose(dataset_id);
   H5Fclose(file_id);
   delete[] data_vec;
}

/**
    Load HCI HDF5 file to a dta buffer.
    @file_name          File name to store
    @dataset_name       HDF5 Dataset name
    @d_type             the type of data buffer
    @array              data buffer to be loaded
*/
template<class TYPE>
void load_hci_hdf5(const char* file_name, 
                   const char* dataset_name, 
                   hid_t d_type, 
                   TYPE* array){

   hid_t       file_id, dataset_id; 

   file_id = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);
   dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);

   H5Dread(dataset_id, d_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
   H5Dclose(dataset_id);
   H5Fclose(file_id);
}

/**
    Save 2D data buffer to HDF5 file.
    Usage: mem2hdf5  ("test.h5", "data", H5T_NATIVE_DOUBLE, w, h, (double *) ptr);
    @file_name          File name to store
    @file_dataset_name  HDF5 Data set name
    @d_type             H5T_NATIVE_INT H5T_NATIVE_CHAR H5T_NATIVE_DOUBLE
    @w                  width
    @h                  height
    @ptr                data buffer pointer
    @img                OpenCV Mat 
*/
template<class TYPE>
void  mem2hdf5      (const char* file_name, 
					 const char* file_dataset_name,
					 hid_t d_type,
                     int w,
                     int h,
                     const TYPE* ptr){

    hid_t       file, dataset,dataset_config; 
    hid_t       datatype, dataspace;   
    hsize_t     dimsf[2];             

    file = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    dimsf[0] = h;
    dimsf[1] = w;

    dataspace = H5Screate_simple(2, dimsf, NULL);

    datatype = H5Tcopy(d_type);
    H5Tset_order(datatype, H5T_ORDER_LE);
    dataset_config = H5Pcreate(H5P_DATASET_CREATE);
    dataset = H5Dcreate2(file, file_dataset_name, datatype, dataspace,
              H5P_DEFAULT, dataset_config, H5P_DEFAULT);

    H5Dwrite(dataset, d_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);

    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Fclose(file);
}


/**
    Load attributes from HCI HDF5 file. It is a hack and only work for spefic parameters!
    @file_name          File name to store
    @shift              attribute name
    @baseline           attribute name
    @focalLength        attribute name
*/
void load_hdf5_attri(const char* file_name, 
                     double& shift, 
                     double& baseline, 
                     double& focalLength){

    hid_t       file_id;  /* identifiers */
    /* Open an existing file. */
    file_id = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);

    hid_t attr_baseline = H5Aopen_by_name( file_id, "/", "dH", H5P_DEFAULT, H5P_DEFAULT );
    if ( attr_baseline >= 0 ) {
        H5Aread( attr_baseline, H5T_NATIVE_DOUBLE, &baseline );
    }
    hid_t attr_focalLength = H5Aopen_by_name( file_id, "/", "focalLength", H5P_DEFAULT, H5P_DEFAULT );
        if ( attr_focalLength >= 0 ) {
            H5Aread( attr_focalLength, H5T_NATIVE_DOUBLE, &focalLength );
    }
    hid_t attr_shift = H5Aopen_by_name( file_id, "/", "shift", H5P_DEFAULT, H5P_DEFAULT );
    if ( attr_shift >= 0 ) {
        H5Aread( attr_shift, H5T_NATIVE_DOUBLE, &shift );
    }
    H5Fclose( file_id );
 }

template<class TYPE>  
void hdf52mat3 ( const char* file_name, 
                const char* file_dataset_name,
                hid_t d_type,
                TYPE,         
                Mat& data){

    hid_t       file_id, dataset_id; 
    file_id      = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);
    dataset_id   = H5Dopen(file_id, file_dataset_name,H5P_DEFAULT);
    hid_t dspace = H5Dget_space(dataset_id);
    const int ndims = H5Sget_simple_extent_ndims(dspace);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dspace, dims, NULL);   
    // Read the dataset. 
    cout<<dims[0]<<" "<<dims[1]<<" "<<dims[2]<<endl;
    cout<<data.rows<<" "<<data.cols<<endl;
    //assert(dims[0] == data.cols);
    //assert(dims[1] == data.rows);
    //assert(dims[2] == 3);
    
    TYPE* data_vec = new TYPE[dims[0]*dims[1]];
    H5Dread(dataset_id, d_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_vec);

    TYPE* temp_ptr = data_vec;
    //cout<<dims[0]<<" "<<dims[1]<<endl;
    for (int j = 0; j <dims[0]; j++){
        for (int i = 0; i < dims[1]; i++)
           data.at<TYPE>(j,i)= *temp_ptr++;
    }

    H5Dclose(dataset_id);
    H5Fclose(file_id);
    delete[] data_vec;

}
#endif

