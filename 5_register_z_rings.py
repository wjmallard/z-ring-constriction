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

from pystackreg import StackReg

from util import load_stack
from util import save_stack
from util import file_exists

MARGIN = 2
MIN_TRACK_LENGTH = 8

def is_too_short(track):
    return len(track) < MIN_TRACK_LENGTH

def is_truncated_at_start(track):
    return track.FRAME.min() <= 0

def is_truncated_at_end(track, t_max):
    return track.FRAME.max() >= t_max

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

    tif_file = basename + '.registered.tif'
    tracks_file = basename + '.TrackSpots.tsv'

    tracks = pd.read_table(tracks_file)
    im = load_stack(tif_file)

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

    # Only the positions are missing on the rows added above. Older pandas
    # skipped the string column here; pandas 3.0 raises on it instead.
    positions = [
        'POSITION_X',
        'POSITION_Y',
        'POSITION_T',
    ]
    tracks[positions] = tracks[positions].interpolate()

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

    t_max, y_max, x_max = np.array(im.shape) - 1

    #%%
    #
    # Extract frames along track.
    #
    sr = StackReg(StackReg.TRANSLATION)

    for (_, track_name), track in tracks.groupby(['TRACK_ID', 'Track_Name']):

        tif_out = f'{basename}.{track_name}.tif'
        if file_exists(tif_out):
            print(f' - {track_name}: Skipping: already processed.')
            continue

        if is_too_short(track):
            print(f' - {track_name}: Skipping: too short.')
            continue
        if is_truncated_at_start(track):
            print(f' - {track_name}: Skipping: truncated by start of timelapse.')
            continue
        if is_truncated_at_end(track, t_max):
            print(f' - {track_name}: Skipping: truncated by end of timelapse.')
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

        crop = np.zeros(shape=(T, Y, X), dtype=im.dtype)

        # Extract frames.
        for _, spot in track.iterrows():

            t_src = spot.FRAME
            t_dst = spot.t_rel
            x1, x2, y1, y2 = spot[['x1', 'x2', 'y1', 'y2']]

            crop[t_dst] = im[t_src,y1:y2,x1:x2]

        # Register.
        tmat = sr.register_stack(crop, reference='previous')
        crop_reg = sr.transform_stack(crop, tmats=tmat)

        # Save to disk.
        save_stack(tif_out, crop_reg)


for n, filename in enumerate(xml_files):
    print(f'[{n+1}/{len(xml_files)}] {filename}')
    isolate_z_rings(filename)

print('Registration complete.')
