**Train**

1. Add your output dir path in the config and choose the model you need (mn, mdsa, cvae, spagan, amgan)
2. Change the dataset path in the dataload/xx.py
3. python train.py

**Test**

1. Put the pretained model in the pre_train dir and change the config
2. python test.py (metrics include SSIM, PSNR and LPIPS)


