#!/usr/bin/env python
# coding: utf-8

# autocontrast for RDS: jpg and tif

# ### 1. Import libararies

# Import libaries
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from skimage.io import imsave
from skimage import exposure

# ### 2. extract image size from .txt file
def extimg(infilename, imgfrac=14/15):
  img_ = plt.imread(infilename)
  h, w = img_.shape
  img = img_[:int(h*imgfrac), :]
  desc = img_[int(h*imgfrac):, :]
  return img, desc

# ### 3. apply autocontrast
def autocontrast(img, desc):
  v_min, v_max = np.percentile(img, (0.2, 99.8))
  img_ac = exposure.rescale_intensity(img, in_range=(v_min, v_max), out_range=(0, 255)).astype(np.uint8)
  img_merge = np.vstack((img_ac, desc))
  return img_merge

# ### 4. save to imagefile
def saveimg(img_ac, infilename):
  dirname = os.path.dirname(infilename)
  basename = os.path.basename(infilename)
  firstname = ".".join(basename.split(".")[:-1])
  outfile_ac = os.path.join(dirname, f"{firstname}_autocontrast.jpg")
  imsave(outfile_ac, img_ac)

# ### Final. Run with arguments
if __name__ == "__main__":
  infilename = sys.argv[1]    # 1. define file name with directory
  img, desc = extimg(infilename)
  img_merge = autocontrast(img, desc)
  saveimg(img_merge, infilename)


