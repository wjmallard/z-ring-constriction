#!/usr/bin/env python
import numpy as np
from aicsimageio import AICSImage

path_dark = '2023.02.18 Fluorescein flats and darks 20pct 100ms/Dark_w1[None].tif'
path_flat = '2023.02.18 Fluorescein flats and darks 20pct 100ms/Flat_w1488 laser 20.tif'
path_out = 'Flat_field_calibration_data.npz'

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
