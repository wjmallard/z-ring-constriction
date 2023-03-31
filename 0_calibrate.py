#!/usr/bin/env python
import numpy as np
from aicsimageio import AICSImage

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

path_dark = '2023.03.29 Fluorescein flats and darks 60x 20pct 100ms/Dark.tif'
path_flat = '2023.03.29 Fluorescein flats and darks 60x 20pct 100ms/Flat.tif'
path_out = '2023.03.xx Flat field calibration data 60x.npz'

dark = AICSImage(path_dark)
flat = AICSImage(path_flat)

dark = dark.data[:,0,0,:,:]
flat = flat.data[:,0,0,:,:]

# Average to get consensus dark and flat frames.
F = flat.mean(axis=0)
D = dark.mean(axis=0)

# Calculate per-pixel gain.
G = (F - D).mean() / (F - D)

# To correct an image:
# C = (im - D) * G

# Write to disk.
np.savez(path_out, Dark=D, Flat=F, Gain=G)
