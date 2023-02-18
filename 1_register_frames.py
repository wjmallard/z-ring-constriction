#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 5 23:23:53 2022

@author: wmallard
"""
import sys
import pathlib
import numpy as np

from aicsimageio.readers import tiff_reader
from aicsimageio.writers import ome_tiff_writer
from pystackreg import StackReg

try:
    filenames = sys.argv[1:]
    assert filenames
except:
    script = sys.argv[0].split('/')[-1]
    print('Usage: %s TIFF_FILE(S)' % script, file=sys.stderr)
    sys.exit(1)

def load_image(filename):
    return tiff_reader.TiffReader(filename, dim_order='TYX')

def save_image(filename, data):
    ome_tiff_writer.OmeTiffWriter.save(data, filename, dim_order='TYX')

def file_exists(filename):
    return pathlib.Path(filename).exists()

def register(filename):

    # Load image.
    im = load_image(filename)
    sr = StackReg(StackReg.TRANSLATION)

    # Generate filenames.
    basename = filename.rsplit('.', 1)[0]
    mat_out = f'{basename}__reg_matrix.npz'
    tif_out = f'{basename}__registered.tif'

    # Register selected channel.
    if file_exists(mat_out):

        print('   - Found a transformation matrix. Loading.')
        tmat = np.load(mat_out)['tmat']

    else:

        print('   - Registering.')
        tmat = sr.register_stack(im.data, reference='previous', verbose=True)

        print('   - Saving transformation matrix.')
        np.savez(mat_out, tmat=tmat)

    # Apply registration to all channels.
    if file_exists(tif_out):
        
        print(f'   - Found a registered tiff file. Skipping.')

    else:

        print(f'   - Applying transformation.')
        reg_img = sr.transform_stack(im.data, tmats=tmat)

        # Save to disk.
        print(f'   - Saving to disk.')
        save_image(tif_out, reg_img)

for n, filename in enumerate(filenames):

    print(f'[{n+1}/{len(filenames)}] {filename}')

    register(filename)
