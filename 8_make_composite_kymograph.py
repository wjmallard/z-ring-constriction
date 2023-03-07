#!/usr/bin/env python
import sys
try:
    tif_files = sys.argv[1:]
    assert tif_files
except:
    script = sys.argv[0].split('/')[-1]
    usage = f'''Usage: {script} TIF_FILE(S)'''
    print(usage, file=sys.stderr)
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import traceback
from glob import glob

from aicsimageio.readers import ome_tiff_reader
from aicsimageio.writers import ome_tiff_writer
from scipy.interpolate import RectBivariateSpline

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

KYMO_WIDTH = 16  # kymograph width on orig image, in pixels
KYMO_RESOLUTION = 100  # kymograph interpolation width, in pixels
MAX_KYMO_DURATION = 100  # max time backwards from t_end

def load_image(filename):
    return ome_tiff_reader.TiffReader(filename, dim_order='TYX').data

def save_image(filename, data):
    ome_tiff_writer.OmeTiffWriter.save(data, filename, dim_order='YX')

def file_exists(filename):
    return pathlib.Path(filename).exists()

def make_line_endpoints(center, theta, length):
    '''
    Generate endpoint xy-coords of a line with the specified parameters.

    Parameters
    ----------
    center : 2-tuple, float
        Coordinates of the center of the line.
    theta : float
        Angle of the line, in degrees, clockwise from the x-axis.
    length : float
        Length, in pixels.

    Returns
    -------
    (float, float), (float, float)
        P1, P2. xy-coords for each end of the generated line.

    '''
    theta = np.radians(theta)

    X = np.array((-length/2, length/2))
    Y = np.zeros(2)

    X_rot = X * np.cos(theta) - Y * np.sin(theta)
    Y_rot = X * np.sin(theta) - Y * np.cos(theta)

    x1, x2 = center[0] + X_rot
    y1, y2 = center[1] + Y_rot

    return (x1, y1), (x2, y2)

def make_kymograph(stack, line, resolution):
    '''
    STACK := an image stack
    LINE := (x1, x2, y1, y2)
    RESOLUTION := number of pixels to interpolate
    '''
    _, nx, ny = stack.shape

    y = np.arange(ny)
    x = np.arange(nx)

    splines = [RectBivariateSpline(y, x, frame) for frame in stack]

    x1, x2, y1, y2 = line

    y = np.linspace(y1, y2, resolution)
    x = np.linspace(x1, x2, resolution)

    interpolated_lines = [sp.ev(y, x) for sp in splines]

    kymograph = np.array(interpolated_lines)

    return kymograph

def find_files_to_composite(tif_files):

    rows = []

    for tif_file in tif_files:

        basename = tif_file[:-len('.tif')]
        ring_posn_file = f'{basename}.division_plane.tsv'
        ring_time_file = f'{basename}.division_params.tsv'

        if not file_exists(ring_posn_file): continue
        if not file_exists(ring_time_file): continue

        df1 = pd.read_table(ring_posn_file)[['cx', 'cy', 'theta']]
        df2 = pd.read_table(ring_time_file)[['t_start', 't_end']]

        df3 = df1.join(df2)
        df3['tif_file'] = tif_file

        rows.append(df3)

    df = pd.concat(rows)
    df = df.reset_index(drop=True)

    return df

def generate_composite(df):

    kymo_stack = np.zeros((len(df), MAX_KYMO_DURATION, KYMO_RESOLUTION), dtype=float)

    for n, row in df.iterrows():

        im = load_image(row.tif_file)

        #
        # Generate kymograph.
        #
        (x1, y1), (x2, y2) = make_line_endpoints((row.cx, row.cy), row.theta, KYMO_WIDTH)
        line = x1, x2, y1, y2

        kymograph = make_kymograph(im, line, KYMO_RESOLUTION)

        #
        # Extract it, flip it, add it to the stack.
        #
        a = row.t_end
        b = max(0, row.t_end - MAX_KYMO_DURATION)
        kymograph = kymograph[a:b:-1]

        #
        # Add it to the stack.
        #
        t_max = min(kymograph.shape[0], MAX_KYMO_DURATION)
        kymo_stack[n,:t_max,:] = kymograph

    composite = kymo_stack.mean(axis=0)[::-1]

    return composite

def construct_filename(tif_files):

    basenames = {filename.rsplit('_s', 1)[0] for filename in tif_files}
    assert len(basenames) == 1
    basename = basenames.pop()
    out_file = f'{basename}.composite.tif'

    return out_file

df = find_files_to_composite(tif_files)
print(f'Found {len(df)} images to composite.')

print('Generating composite.')
composite = generate_composite(df)
print('Done.')

print('Writing to disk.')
out_file = construct_filename(tif_files)
save_image(out_file, composite)

print('Composite complete.')
