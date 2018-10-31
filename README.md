# Light-Field-Reconstruction

![sample](https://github.com/Computational-Camera/Light_Field_Reconstruction/blob/master/img/intro.jpg)

This project contains implementations paper titled ["Multi-view Image Restoration From Plenoptic Raw Images" ](https://link.springer.com/chapter/10.1007/978-3-319-16631-5_1) ACCV 2014, Emerging Topics In Image Restoration and Enhancement Workshop.

We introduce a novel technique to reconstruct the 4D light field from the microlens-array-based light-field camera without the need for the reference image using a simple and efficient blind global grid fitting approach. In decoding the 4D light field from a 2D raw image, the centre of each microlens image is regarded as the origin of the angular samples inside the microlens image. However, the microlens array is not perfectly aligned with the sensor due to manufacturing and assembly imperfections. Accurately detecting the centre of each microlens image is the fundamental requirement for restoring high-quality light fields.

If you use this code/model for your research, please cite the following paper:
```
@inproceedings{multi-view_restoration_accv_2014,
    author = { Shan Xu, Zhi-Liang Zhou and Nicholas Devaney},
    title  = {Multi-view Image Restoration},
    booktitle = {ACCV 2014 Emerging Topics In Image Restoration and Enhancement Workshop},
    year   = {2014}
}
```
Dataset can be obtained from this  [link](https://www.dropbox.com/s/gxgvh5pywe99xsc/Light-Field-Reconstruction.zip)


Besides the approach mentioned in the paper, we also implement a function to estimate the diameter of the microlens.  As shown in the figure below, the regular geometric structure of the hexagonal microlens array results in six high peaks distributed uniformly around the zero frequency. Although the spectrum of the content image is contaminated by the content changes inside each microlens image, the six high peak frequency components are preserved and can be used to determine the size of the hexagonal grid.

![sample](https://github.com/Computational-Camera/Light_Field_Reconstruction/blob/master/img/ml_fft.jpg)


### Building on Ubuntu
The code depends OpenCV library (Version>3.0).
Use qmake to generate the makefile
```
./qmake
```
Compile the code
```
make -j4
```





