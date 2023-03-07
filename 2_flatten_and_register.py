#!/usr/bin/env python
import sys

try:
    calib_npz = sys.argv[1]
    filenames = sys.argv[2:]
    assert calib_npz.endswith('.npz')
    assert filenames
except:
    script = sys.argv[0].split('/')[-1]
    usage = f'''Usage: {script} CALIBRATION_NPZ TIFF_FILE(S)
    
    CALIBRATION_NPZ contains dark and flat field composites.
    '''
    print(usage, file=sys.stderr)
    sys.exit(1)

import pathlib
import numpy as np

from aicsimageio.readers import tiff_reader
from aicsimageio.writers import ome_tiff_writer
from pystackreg import StackReg

def load_image(filename):
    return tiff_reader.TiffReader(filename, dim_order='TYX').data

def save_image(filename, data):
    ome_tiff_writer.OmeTiffWriter.save(data, filename, dim_order='TYX')

def file_exists(filename):
    return pathlib.Path(filename).exists()

def load_calibration(calib_npz):
    '''
    Load the dark and flat field calibration matrices.
    Generate a gain matrix.
    Leave the dark field and gain matrices into the global namespace.
    '''
    global Dark
    global Gain

    calib = np.load(calib_npz)
    Dark = calib['Dark']
    Flat = calib['Flat']
    Gain = (Flat - Dark).mean() / (Flat - Dark)

def flatten(im):
    '''
    Note: We assume the dark field and gain matrices
    have already been loaded into the global namespace.
    '''
    global Dark
    global Gain

    # Flatten image.
    print('   - Flattening.')
    Flat = (im - Dark) * Gain
    
    return Flat

def register(im):

    sr = StackReg(StackReg.TRANSLATION)

    print('   - Registering.')
    tmat = sr.register_stack(im, reference='previous', verbose=True)

    print(f'   - Applying transformation.')
    reg_img = sr.transform_stack(im, tmats=tmat)

    return reg_img

def cast_to_uint16(im):
    print('   - Casting to uint16.')
    im = im.round()
    im = np.clip(im, 0, 2**16-1)
    im = im.astype(np.uint16)
    return im


load_calibration(calib_npz)

for n, filename in enumerate(filenames):

    print(f'[{n+1}/{len(filenames)}] {filename}')

    # Process image.
    im = load_image(filename)
    im = flatten(im)
    im = register(im)
    im = cast_to_uint16(im)
    
    # Save to disk.
    basename = filename.rsplit('.', 1)[0]
    tif_out = f'{basename}.registered.tif'
    save_image(tif_out, im)
