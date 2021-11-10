from ij import IJ, ImagePlus, ImageStack
from ij.process import FloatProcessor
from ij.plugin import RGBStackMerge, RGBStackConverter

import os

# RGB thresholds
th = [192, 100, 100] # R, G, B

# files
workDir = r"C:\Arbeitplatz\03_ImageJ_script_learning\sample_code\images"
#srcFile = "blobs_rgb.tif"
srcFile = "Lenna.png"
dstFile = srcFile[:-4] + "_out.tif"

srcpath = os.path.join(workDir, srcFile)
dstpath = os.path.join(workDir, dstFile)

# function: thresholding
def threshold(p):
	if p > th[i]: 
		return p
	else: 
		return 0.0

# open image (RGB)
imp = IJ.openImage(srcpath)

# imageProcessor
ip = imp.getProcessor()

# FloatProcessor by channel
chpx = []         # channel pixels
filtered = []     # filtered FloatProcessor
for i in range(3):
	chpx.append(ip.toFloat(i, None).getPixels())

	_filtered = [threshold(p) for p in chpx[i]] # filtered pixels
	filtered.append(FloatProcessor(ip.width, ip.height, _filtered, None))

# save each channels
for i in range(3):
	IJ.save(ImagePlus("filtered_%d" %i, filtered[i]), dstpath[:-4] + '_%d' %i)

# add slices to new stack
stack_new = ImageStack(imp.width, imp.height)

for i in range(3):
	stack_new.addSlice(None, filtered[i])

# new image (stack)
imp_new = ImagePlus("new", stack_new)
imp_new.show()	
IJ.save(imp_new, dstpath[:-4] + '_stack.tif')

# new image (color)
mergeimp = RGBStackMerge.mergeChannels([ImagePlus(None,filtered[0]), ImagePlus(None,filtered[1]), ImagePlus(None,filtered[2]), None, None, None, None], False)
mergeimp.title = "filtered"
RGBStackConverter.convertToRGB(mergeimp)
mergeimp.show()
IJ.save(mergeimp, dstpath)



