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

    for n, (_, row) in enumerate(df.iterrows()):

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

    if len(df) == 0:
        print(f' - Warning: No tracks found for {xml_file}')
        return None

    ring_data = pd.concat(rows)

    #
    # Merge into main table.
    #
    df = pd.merge(df, ring_data, how='left', on='Track_Name')
    df = df.drop(['ring_posn_tsv', 'ring_time_tsv'], axis=1)

    return df

def extract_basename(xml_file):

    basename = xml_file[:-len('.xml')]

    if '_s' in basename:
        return basename.rsplit('_s', 1)[0]
    else:
        return basename

def extract_field(xml_file):
    if '_s' not in xml_file: return 1
    return int(xml_file.rsplit('_s', 1)[1].split('.', 1)[0])

def extract_replicate(xml_file):
    return int(xml_file.rsplit(' rep', 1)[1].split('_', 1)[0])

def compile_track_info(xml_files):

    print(f'Found {len(xml_files)} TrackMate runs.')

    dfs = []

    for xml_file in xml_files:

        df = load_track_info(xml_file)
        if df is None: continue

        df.index.name = 'n'
        df = df.reset_index()

        df['xml_file'] = xml_file
        df['basename'] = df.xml_file.apply(extract_basename)
        df['replicate'] = df.tif_file.apply(extract_replicate)
        df['field'] = df.tif_file.apply(extract_field)

        dfs.append(df)

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)

    df['num_fields'] = df.groupby(['basename', 'replicate']).field.transform(lambda s: s.nunique())

    print(f'Found {len(df)} images in total.')
    print()

    return df

def write_metadata_tsv(outfile, df):

    num_images = len(df)
    num_fields = df.iloc[0].num_fields

    df = pd.DataFrame({
        'num_images': [num_images],
        'num_fields': [num_fields],
    })
    df.to_csv(out_file, sep='\t', index=None)

#
# Main()
#
experiments = compile_track_info(xml_files)

for (basename, replicate), df in experiments.groupby(['basename', 'replicate']):

    print(f'Processing: {basename}')
    print(f' - {len(df)} images from {df.iloc[0].num_fields} fields')

    print(' - Generating composite.')
    composite = generate_composite(df)

    print(' - Writing to disk.')
    out_file = df.iloc[0].basename + '.composite.tif'
    save_image(out_file, composite)

    out_file = df.iloc[0].basename + '.composite.tsv'
    write_metadata_tsv(out_file, df)

print()
print('Composite complete.')
