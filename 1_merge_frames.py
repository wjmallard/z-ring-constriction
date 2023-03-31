#!/usr/bin/env python
import sys
try:
    if sys.argv[1] == '--all':
        MERGE_FIELDS = True
        directories = sys.argv[2:]
    else:
        MERGE_FIELDS = False
        directories = sys.argv[1:]
    assert directories
except:
    script = sys.argv[0].split('/')[-1]
    print(f'Usage: {script} [--all] DIRECTORIES', file=sys.stderr)
    print( '  Use "--all" to merge all fields into a single tiff.')
    print( '  This is intended for flat field correction stacks.')
    sys.exit(1)

import numpy as np
import pandas as pd
from glob import glob
import os

from util import load_image
from util import save_stack
from util import dir_exists

def extract_sample(filename):
    return filename.split('_w')[0]

def extract_timestep(filename):
    return int(filename.rsplit('_t')[1].rsplit('.', 1)[0])

def extract_series(filename):
    return int(filename.rsplit('_s')[1].rsplit('_', 1)[0])

def generate_file_list(directory):

    directory = os.path.abspath(directory)
    paths = glob(directory + '/*.TIF')

    df = pd.DataFrame(paths, columns=['Path'])
    df['Directory'] = df.Path.str.rsplit('/', 1).str[0]
    df['Filename'] = df.Path.str.rsplit('/', 1).str[1]
    df['Parent'] = df.Directory.str.rsplit('/', 1).str[0]

    df['Sample'] = df.Filename.apply(extract_sample)
    df['Field'] = 1
    df['Timestep'] = 1

    df['is_timeseries'] = df.Filename.str.contains('_t')
    df['is_multipoint'] = df.Filename.str.contains('_s')

    df.loc[df.is_multipoint, 'Field'] = df[df.is_multipoint].Filename.apply(extract_series)
    df.loc[df.is_timeseries, 'Timestep'] = df[df.is_timeseries].Filename.apply(extract_timestep)

    df['Path_Out'] = ''
    if MERGE_FIELDS:
        df.loc[:, 'Path_Out'] = df.Parent + '/' + df.Sample + '.tif'
    else:
        x = df[df.is_multipoint]
        df.loc[df.is_multipoint, 'Path_Out'] = x.Parent + '/' + x.Sample + '_s' + x.Field.astype(str) + '.tif'

        x = df[~df.is_multipoint]
        df.loc[~df.is_multipoint, 'Path_Out'] = x.Parent + '/' + x.Sample + '.tif'

    df = df.sort_values(['Sample', 'Field', 'Timestep'])
    df = df.reset_index(drop=True)

    return df

for n, directory in enumerate(directories):

    print(f'[{n+1}/{len(directories)}] {directory}')

    if not dir_exists(directory):
        print(' - Not a directory. Skipping.')
        continue

    files = generate_file_list(directory)

    if len(files) == 0:
        print(' - No images found. Skipping.')
        continue

    for path_out, df in files.groupby('Path_Out'):

        file_out = path_out.rsplit('/', 1)[-1]

        print(f' - {file_out}')
        im = np.stack([load_image(f) for f in df.Path])
        save_stack(path_out, im)
