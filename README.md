# pytorch-LDRN
Title:Lightweight Dual-stream ResidualNetwork for Single ImageSuper-Resolution
By Yichun Jiang, Yunqing Liu, Weida Zhang, Depeng Zhu

School of Electronic Information Engineering,Changchun University of Science and Technology,Jilin 130022,China
Changchun University of Science and Technology National Demonstration Center for Experimental Electrical, ChangChun, JiLin, 130022, China

Dear experts, scholars and related practitioners:：

This is our code for new algorithm called LDRN, which is proposed to solve the super-resolution problem.

We have given our model here, which can be used to verify our results.
However, due to github restrictions, we saved the test datasets on OneDrive. You can download the test datasets from the following URL: 

https://1drv.ms/u/s!AsnHFFsP0cnXkDl82thlis6dXlLT?e=JTo3BC

Unzip the datasets and copy them to the root directory. Then, you can run  test.py to generate super-resolution results.
If you want to train our model by yourself, please download the training data set yourself and run main.py.

All of our experiments were conducted using Pytorch1.7 on a NVIDIA RTX3090 GPU. 
|     Algorithm          |     Scale    |     Params(K)    |     Multi-Adds(G)    |     Set5     PSNR/SSIM    |     Set14     PSNR/SSIM    |     BSD100     PSNR/SSIM    |     Urban100     PSNR/SSIM    |
|------------------------|--------------|------------------|----------------------|---------------------------|----------------------------|-----------------------------|-------------------------------|
|     Bicubic            |     4x       |     -            |     -                |     28.43/0.802           |     26.10/0.694            |     25.96/0.660             |     23.15/0.659               |
|     A+                 |     4x       |     -            |     -                |     30.33/0.856           |     27.44/0.745            |     26.83/0.700             |     24.34/0.721               |
|     SRCNN(2016)        |     4x       |     57           |     52.7             |     30.48/0.863           |     27.49/0.750            |     26.90/0.710             |     24.52/0.726               |
|     FSRCNN(2016)       |     4x       |     12           |     4.6              |     30.71/0.866           |     27.60/0.752            |     26.97/0.715             |     24.62/0.728               |
|     VDSR(2016)         |     4x       |     665          |     612.6            |     31.35/0.884           |     28.01/0.767            |     27.29/0.725             |     25.18/0.752               |
|     LapSRN(2017)       |     4x       |     813          |     149.4            |     31.54/0.885           |     28.19/0.772            |     27.32/0.728             |     25.21/0.756               |
|     MemNet(2017)       |     4x       |     677          |     623.9            |     31.74/0.889           |     28.26/0.772            |     27.40/0.728             |     25.50/0.763               |
|     IDN(2018)          |     4x       |     600          |     34.5             |     31.82/0.890           |     28.25/0.773            |     27.41/0.730             |     25.41/0.763               |
|     CARN-M(2018)       |     4x       |     412          |     32.5             |     31.92/0.890           |     28.42/0.776            |     27.44/0.730             |     25.62/0.769               |
|     MADNet-LF(2020)    |     4x       |     1002         |     54.1             |     32.05/0.892           |     28.45/0.778            |     27.47/0.734             |     25.77/0.775               |
|     s-LWSR32(2020)     |     4x       |     571          |     37.3             |     32.04/0.893           |     28.15/0.776            |     27.50/0.734             |     25.87/0.779               |
|     LDRN(ous)          |     4x       |     879          |     47.8             |     32.00/0.893           |     28.10/0.780            |     27.57/0.738             |     25.92/0.782               |
