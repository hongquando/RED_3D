**High resolution 3D image with Regularization by Denoising (RED 3D)**

* Integration RED 3D with different denoising methods
    - Median filter
    - Non-local means
    - Wavelet filter
    - BM4D
    - FastDVDNet
* With Median filter, Non-local means, Wavelet filter, BM4D

* For Median:
```
python3 main.py \
--image_path ./shepp_logan_128.mat \
--input_path ./input_median/ \
--output_path ./output_median/ \
--save_output 5 \
--psf_sz 9 \
--gaussian_std 3 \
--BSNRdb 10 \
--max_iter 101 \
--lam 0.02 \
--use_parallel \
--denoiser median
```
* For Non-local means:
```
python3 main.py \
--image_path ./shepp_logan_128.mat \
--input_path ./input_nlm/ \
--output_path ./output_nlm/ \
--save_output 5 \
--psf_sz 9 \
--gaussian_std 3 \
--BSNRdb 10 \
--max_iter 101 \
--lam 0.02 \
--use_parallel \
--denoiser nlm \
--sigma 0.1 
```
* For Wavelet filter:
```
python3 main.py \
--image_path ./shepp_logan_128.mat \
--input_path ./input_wl/ \
--output_path ./output_wl/ \
--save_output 5 \
--psf_sz 9 \
--gaussian_std 3 \
--BSNRdb 10 \
--max_iter 101 \
--lam 0.02 \
--use_parallel \
--denoiser wavelet \
--lmax 3
```
* For BM4D:
```
python3 main.py \
--image_path ./shepp_logan_128.mat \
--input_path ./input_bm4d/ \
--output_path ./output_bm4d/ \
--save_output 5 \
--psf_sz 9 \
--gaussian_std 3 \
--BSNRdb 10 \
--max_iter 101 \
--lam 0.02 \
--use_parallel \
--denoiser bm4d \
--sigma 0.1
```
* With FastDVDNet
```
python3 main.py \
--image_path ./shepp_logan_128.mat \
--input_path ./input_fast/ \
--output_path ./output_fast/ \
--save_output 5 \
--psf_sz 9 \
--gaussian_std 3 \
--BSNRdb 10 \
--max_iter 101 \
--lam 0.02 \
--use_parallel \
--denoiser fastdvdnet \
--sigma 0.1 \
--model_file ./net.pth
```

**NOTES**
* run with *--help* to see details on all input parameters
* If you run code with param ```--crop```, you need to add the code below before calcul PSNR between ground truth image and input image
```
from utils import crop_image

gt_mat = scipy.io.loadmat("./shepp_logan_64.mat")
gt_image = gt_mat[list(gt_mat.keys())[-1]]
gt_image,_,_,_ = crop_image(gt_image)

```
