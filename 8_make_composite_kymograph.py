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

from util import load_stack
from util import save_image
from util import file_exists

from kymo import make_kymograph
from kymo import make_kymograph_grid

KYMO_WIDTH = 12  # kymograph width on orig image, in pixels
KYMO_RESOLUTION = 8  # Number of points to interpolate per pixel.
MAX_KYMO_DURATION = 40  # max time backwards from t_end

HISTOGRAM_BINS = 15
HISTOGRAM_XMAX = 150
HISTOGRAM_YMAX = 100

def straighten_kymograph(kymograph, npz_file):

    kymo_straight = np.zeros_like(kymograph)
    npz = np.load(npz_file)

    center = KYMO_WIDTH * KYMO_RESOLUTION / 2
    peak_locs = npz['fwhm_loc']

    offsets = np.round(center - peak_locs).astype(int)

    for i, (row, offset) in enumerate(zip(kymograph, offsets)):
        a = max(0, offset)
        b = min(len(row), len(row) + offset)
        c = max(0, -offset)
        d = min(len(row), len(row) - offset)
        kymo_straight[i,a:b] = row[c:d]

    return kymo_straight

def generate_constriction_composite(df):

    kymo_stack = np.zeros((len(df), MAX_KYMO_DURATION, KYMO_WIDTH * KYMO_RESOLUTION), dtype=float)

    for n, (_, row) in enumerate(df.iterrows()):

        im = load_stack(row.tif_file)

        #
        # Generate kymograph.
        #
        kymograph = make_kymograph(im, row.cx, row.cy, row.theta, KYMO_WIDTH, KYMO_RESOLUTION)

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

def generate_condensation_composite(df, width, w_res):

    kymo_stack = np.zeros((len(df), MAX_KYMO_DURATION, KYMO_WIDTH * KYMO_RESOLUTION), dtype=float)

    for n, (_, row) in enumerate(df.iterrows()):

        im = load_stack(row.tif_file)

        #
        # Generate kymograph.
        #
        kymograph = make_kymograph_grid(im, row.cx, row.cy, row.theta, KYMO_WIDTH, KYMO_RESOLUTION, width, w_res)

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
    df['t_end_absolute'] = df.track_start + df.t_end

    print(f'Found {len(df)} images in total.')
    print()

    return df

def save_png(outfile, im, df):

    num_images = len(df)
    num_fields = df.iloc[0].num_fields

    plt.close('all')

    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    ax.imshow(im, cmap='Greys_r')

    msg = f'N={num_images}'
    ax.text(.01, .01, msg,
            color='white',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes)

    fig.canvas.print_png(outfile)

def save_QC_plot(out_file, comp_cons, comp_cond, df):

    # Configure plots.
    fig, axes = plt.subplots(1, 3, squeeze=False)
    fig.set_figheight(5 * axes.shape[0])
    fig.set_figwidth(5 * axes.shape[1])

    num_images = len(df)
    msg = f'N={num_images}'

    # Ring constriction kymograph:
    ax = axes[0,0]
    ax.imshow(comp_cons, cmap='Greys_r')
    ax.text(.01, .01, msg,
            color='white',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes)

    # Ring condensation kymograph:
    ax = axes[0,1]
    ax.imshow(comp_cond, cmap='Greys_r')
    ax.text(.01, .01, msg,
            color='white',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes)

    # Histogram of constriction end times:
    ax = axes[0,2]
    ax.hist(df.t_end_absolute,
            range=(0, HISTOGRAM_XMAX),
            bins=HISTOGRAM_BINS)
    ax.set_xlim(0, HISTOGRAM_XMAX)
    ax.set_ylim(0, HISTOGRAM_YMAX)
    ax.set_xlabel('End of constriction')
    ax.set_ylabel('Count')

    # Save to disk.
    fig.tight_layout()
    fig.canvas.print_png(out_file)

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

# t_min, t_max = 0, 100
# t_min, t_max = 100, 200
# t_min, t_max = 200, 300
# experiments = experiments[experiments.t_end_absolute > t_min]
# experiments = experiments[experiments.t_end_absolute <= t_max]
# subset = f'.{t_min}_to_{t_max}'
subset = ''

for (basename, replicate), df in experiments.groupby(['basename', 'replicate']):

    print(f'Processing: {basename}')
    print(f' - {len(df)} images from {df.iloc[0].num_fields} fields')

    # Generate ring constriction kymograph.
    print(' - Generate composite ring constriction kymograph.')
    comp_cons = generate_constriction_composite(df)

    print(' - Writing to disk.')
    out_file = df.iloc[0].basename + subset + '.composite_constriction.tif'
    save_image(out_file, comp_cons)

    out_file = df.iloc[0].basename + subset + '.composite_constriction.png'
    save_png(out_file, comp_cons, df)

    out_file = df.iloc[0].basename + subset + '.composite.tsv'
    write_metadata_tsv(out_file, df)

    # Generate ring condensation kymograph.
    df.theta += 90

    print(' - Generating composite ring condensation kymograph.')
    width = 8
    w_res = 2
    comp_cond = generate_condensation_composite(df, width, w_res)

    print(' - Writing to disk.')
    out_file = df.iloc[0].basename + subset + '.composite_condensation.tif'
    save_image(out_file, comp_cond)

    out_file = df.iloc[0].basename + subset + '.composite_condensation.png'
    save_png(out_file, comp_cond, df)

    # Generate QC plot.
    out_file = df.iloc[0].basename + subset + '.composite_QC.png'
    save_QC_plot(out_file, comp_cons, comp_cond, df)

print()
print('Composite complete.')
