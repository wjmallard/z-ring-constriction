#!/usr/bin/env python
import sys
try:
    xml_files = sys.argv[1:]
    assert xml_files
except:
    script = sys.argv[0].split('/')[-1]
    usage = f'''Usage: {script} TrackMate.xml'''
    print(usage, file=sys.stderr)
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    return os.path.isfile(filename)

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

def straighten_kymograph(kymograph, npz_file):

    kymo_straight = np.zeros_like(kymograph)
    npz = np.load(npz_file)

    center = KYMO_RESOLUTION / 2
    peak_locs = npz['fwhm_loc']

    offsets = np.round(center - peak_locs).astype(int)

    for i, (row, offset) in enumerate(zip(kymograph, offsets)):
        a = max(0, offset)
        b = min(len(row), len(row) + offset)
        c = max(0, -offset)
        d = min(len(row), len(row) - offset)
        kymo_straight[i,a:b] = row[c:d]

    return kymo_straight

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
        # Straighten kymograph.
        #
        kymograph = straighten_kymograph(kymograph, row.ring_time_npz)

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

def load_track_info(xml_file):

    basename = xml_file[:-len('.xml')]

    tracks_tsv = f'{basename}.TrackMetadata.tsv'

    df = pd.read_table(tracks_tsv)
    df = df[['Track_Name']]

    #
    # Construct relevant filenames.
    #
    df['tif_file'] = basename + '.' + df.Track_Name + '.tif'
    df['ring_posn_tsv'] = basename + '.' + df.Track_Name +  '.ring_position.tsv'
    df['ring_time_tsv'] = basename + '.' + df.Track_Name +  '.ring_timing.tsv'
    df['ring_time_npz'] = basename + '.' + df.Track_Name +  '.ring_timing.npz'
    df['composite_tif'] = basename + '.composite.tif'

    #
    # Skip tracks where finding the ring position or constriction time failed.
    #
    df = df[df.ring_posn_tsv.apply(file_exists)]
    df = df[df.ring_time_tsv.apply(file_exists)]
    df = df[df.ring_time_npz.apply(file_exists)]
    df = df.reset_index(drop=True)

    #
    # Extract ring position and constriction timing info from tsv files.
    #
    rows = []

    for _, row in df.iterrows():

        posn_data = pd.read_table(row.ring_posn_tsv)[['cx', 'cy', 'theta']]
        time_data = pd.read_table(row.ring_time_tsv)[['t_start', 't_end', 'track_start']]

        merged_data = posn_data.join(time_data)
        merged_data['Track_Name'] = row.Track_Name

        rows.append(merged_data)

    ring_data = pd.concat(rows)

    #
    # Merge into main table.
    #
    df = pd.merge(df, ring_data, how='left', on='Track_Name')
    df = df.drop(['ring_posn_tsv', 'ring_time_tsv'], axis=1)

    return df

xml_file = xml_files[0]

df = load_track_info(xml_file)
print(f'Found {len(df)} images to composite.')

print('Generating composite.')
composite = generate_composite(df)
print('Done.')

print('Writing to disk.')
out_file = df.iloc[0].composite_tif
save_image(out_file, composite)

print('Composite complete.')
