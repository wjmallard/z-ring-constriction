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

from aicsimageio.readers import TiffReader
from aicsimageio.writers import ome_tiff_writer
from pystackreg import StackReg

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MARGIN = 2
MIN_TRACK_LENGTH = 20

def save_image(filename, data):
    ome_tiff_writer.OmeTiffWriter.save(data, filename, dim_order='TYX')

def is_out_of_bounds(track, x_max, y_max):
    return ((track.x1 < 0).any() |
            (track.x2 >= x_max).any() |
            (track.y1 < 0).any() |
            (track.y2 >= y_max).any())

def isolate_z_rings(xml_file):

    #
    # Load tracks and image.
    #
    basename = xml_file[:-len('.xml')]

    tif_file = basename + '.tif'
    tracks_file = basename + '.TrackSpots.tsv'

    tracks = pd.read_table(tracks_file)
    im = TiffReader(tif_file)

    radius = tracks.RADIUS.median()

    #
    # Drop extraneous columns.
    #
    tracks = tracks[[
        'Track_Name',
        'TRACK_ID',
        'FRAME',
        'POSITION_X',
        'POSITION_Y',
        'POSITION_T',
    ]]

    #%%
    #
    # Interpolate missing frames.
    #
    interpolated_points = []

    for track_name, track in tracks.groupby('Track_Name'):

        track_id = track.iloc[0].TRACK_ID

        full_range = np.arange(track.FRAME.min(), track.FRAME.max() + 1)
        missing_frames = set(full_range) - set(track.FRAME)

        for frame in missing_frames:
            interpolated_points.append((track_name, track_id, frame))

    new_rows = pd.DataFrame(data=interpolated_points, columns=['Track_Name', 'TRACK_ID', 'FRAME'])

    tracks = pd.concat([tracks, new_rows])
    tracks = tracks.sort_values(['TRACK_ID', 'FRAME'])
    tracks = tracks.reset_index(drop=True)

    tracks = tracks.interpolate()

    #%%
    #
    # Generate extraction coordinates.
    #
    tracks['t_rel'] = tracks.groupby('Track_Name').cumcount()

    dXY = MARGIN * radius

    tracks['x1'] = (tracks.POSITION_X - dXY).astype(int)
    tracks['x2'] = (tracks.POSITION_X + dXY).astype(int)
    tracks['y1'] = (tracks.POSITION_Y - dXY).astype(int)
    tracks['y2'] = (tracks.POSITION_Y + dXY).astype(int)

    t_max, y_max, x_max = im.shape

    #%%
    #
    # Extract frames along track.
    #
    sr = StackReg(StackReg.TRANSLATION)

    for (_, track_name), track in tracks.groupby(['TRACK_ID', 'Track_Name']):

        if len(track) < MIN_TRACK_LENGTH:
            print(f' - {track_name}: Skipping: too short.')
            continue
        elif is_out_of_bounds(track, x_max, y_max):
            print(f' - {track_name}: Skipping: too close to edge.')
            continue
        else:
            print(f' - {track_name}: Registering {len(track)} frames.')

        # Initialize output tiff stack.
        T = len(track)
        X = np.max(track.x2 - track.x1)
        Y = np.max(track.y2 - track.y1)

        crop = np.zeros(shape=(T, X, Y), dtype=im.data.dtype)

        # Extract frames.
        for _, spot in track.iterrows():

            t_src = spot.FRAME
            t_dst = spot.t_rel
            x1, x2, y1, y2 = spot[['x1', 'x2', 'y1', 'y2']]

            crop[t_dst] = im.data[t_src,y1:y2,x1:x2]

        # Register.
        tmat = sr.register_stack(crop, reference='previous')
        crop_reg = sr.transform_stack(crop, tmats=tmat)

        # Save to disk.
        tif_out = f'{basename}.{track_name}.tif'
        save_image(tif_out, crop_reg)


for n, filename in enumerate(xml_files):
    print(f'[{n+1}/{len(xml_files)}] {filename}')
    isolate_z_rings(filename)

print('Registration complete.')
