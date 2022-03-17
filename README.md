
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
--image_path ./shepp_logan_256.mat \
--input_path ./input/ \
--output_path ./output/ \
--save_output 5 \
--crop \
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
--image_path ./shepp_logan_256.mat \
--input_path ./input/ \
--output_path ./output/ \
--save_output 5 \
--crop \
--psf_sz 9 \
--gaussian_std 3 \
--BSNRdb 10 \
--max_iter 101 \
--lam 0.02 \
--use_parallel \
--denoiser nlm \
--sigma 0.1 \
```
* For Wavelet filter:
```
python3 main.py \
--image_path ./shepp_logan_256.mat \
--input_path ./input/ \
--output_path ./output/ \
--save_output 5 \
--crop \
--psf_sz 9 \
--gaussian_std 3 \
--BSNRdb 10 \
--max_iter 101 \
--lam 0.02 \
--use_parallel \
--denoiser wavelet \
--lmax 3
```
* With FastDVDNet
```
python3 main.py \
--image_path ./shepp_logan_256.mat \
--input_path ./input/ \
--output_path ./output/ \
--save_output 5 \
--crop \
--psf_sz 9 \
--gaussian_std 3 \
--BSNRdb 10 \
--max_iter 101 \
--lam 0.02 \
--use_parallel \
--denoiser fastdvdne \
--sigma 0.1 \
--model_file ./net.pth
```

**NOTES**
* run with *--help* to see details on all input parameters

