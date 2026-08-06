#!/usr/bin/env python
import sys
try:
    tif_files = sys.argv[1:]
    assert tif_files
except:
    script = sys.argv[0].split('/')[-1]
    usage = f'''Usage: {script} COMPOSITE_TIFF(S)

    Accepts the composite kymographs written by 8_make_composite_kymograph.py,
    either .composite_constriction.tif or .composite_condensation.tif. (input)

    Measures ring width over time by running the same FWHM analysis used on
    individual rings, and converts pixels to microns. Composites are grouped
    by strain, so pass every replicate of a strain in one invocation.

    Produces, per strain and per kymograph type:
      <strain>.<type>_curves.png   one trace per replicate, plus mean +/- std
      <strain>.<type>_curves.tsv   the same curves as numbers

    Replicates with too few rings are skipped.
    '''
    print(usage, file=sys.stderr)
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

from util import load_image
from util import file_exists

from fwhm import find_kymograph_peaks

KYMO_RESOLUTION = 8  # Number of points to interpolate per pixel.

NM_PER_PIXEL = 224  # measured with a stage micrometer.
TO_MICRONS = NM_PER_PIXEL / 1e3 / KYMO_RESOLUTION

MIN_RINGS_PER_COMPOSITE = 30  # Min rings for a replicate to be plotted.

# Y-axis limits in microns, per kymograph type. A type listed here is held
# to fixed limits so figures are comparable across strains; anything else
# is autoscaled.
YLIM = {
    'condensation': (.64, 1.15),
    'constriction': (.60, 1.15),
}

def extract_strain(tif_file):
    '''
    Strain name is the first word of the filename, eg "bWM204".
    '''
    return tif_file.rsplit('/', 1)[-1].split()[0]

def extract_replicate(tif_file):
    '''
    Replicate label, eg "2023.03.27 rep2".

    The date is required: each imaging session numbers its replicates from
    one, so "rep2" alone collides across sessions of the same strain.
    '''
    date = tif_file.rsplit('/', 2)[-2].split()[0]
    rep = tif_file.rsplit('rep', 1)[1].split('.', 1)[0]

    return f'{date} rep{rep}'

def extract_kind(tif_file):
    '''
    Kymograph type, eg "constriction" or "condensation".
    '''
    return tif_file.rsplit('.composite_', 1)[1].rsplit('.tif', 1)[0]

def count_rings(tif_file):
    '''
    Read the ring count from the sibling .composite.tsv.
    '''
    tsv_file = tif_file.rsplit('.composite_', 1)[0] + '.composite.tsv'

    if not file_exists(tsv_file):
        return None

    df = pd.read_table(tsv_file)
    num_rings = int(df.num_images.iloc[0])

    return num_rings

def measure_width(tif_file):
    '''
    Measure ring width over time, in microns.

    Runs the same FWHM analysis used on individual rings.
    '''
    kymograph = load_image(tif_file)

    results = find_kymograph_peaks(kymograph)
    fwhm_width = results[1]

    return fwhm_width * TO_MICRONS

def make_figure(strain, kind, curves):
    '''
    Plot one trace per replicate, plus the mean +/- half a standard deviation.
    '''
    basename = f'{strain}.{kind}_curves'

    plt.figure(figsize=(12, 5))

    for rep, num_rings, width in curves:
        plt.plot(width, label=f'{rep} [n={num_rings}]')

    widths = np.array([width for _, _, width in curves])
    mean = widths.mean(axis=0)
    std = widths.std(axis=0)
    x = np.arange(len(mean))

    plt.plot(mean, 'k:', label='mean')
    plt.fill_between(
        x,
        mean - std / 2,
        mean + std / 2,
        color='gray',
        alpha=.1,
    )

    if kind in YLIM:
        plt.ylim(*YLIM[kind])

    plt.title(f'{strain} -- {kind}')
    plt.xlabel('Time [minutes]')
    plt.ylabel('Width [um]')
    plt.legend(loc='lower left')

    plt.tight_layout()

    png_file = f'{basename}.png'
    print(f' - {png_file}')
    plt.savefig(png_file)
    plt.close('all')

    #
    # Save curve data to tsv.
    #
    out = {'time': x}
    for rep, _, width in curves:
        out[rep] = width
    out['mean'] = mean
    out['std'] = std

    tsv_file = f'{basename}.tsv'
    print(f' - {tsv_file}')
    pd.DataFrame(out).to_csv(tsv_file, sep='\t', index=None)

#
# Main()
#
groups = defaultdict(list)

for tif_file in tif_files:

    num_rings = count_rings(tif_file)

    if num_rings is None:
        print(f'Missing composite.tsv. Skipping: {tif_file}')
        continue

    if num_rings < MIN_RINGS_PER_COMPOSITE:
        print(f'Only {num_rings} rings. Skipping: {tif_file}')
        continue

    key = (extract_strain(tif_file), extract_kind(tif_file))
    groups[key].append((extract_replicate(tif_file), num_rings, measure_width(tif_file)))

for (strain, kind), curves in sorted(groups.items()):

    curves = sorted(curves)

    print(f'{strain} -- {kind}: {len(curves)} replicates')

    lengths = {len(width) for _, _, width in curves}
    if len(lengths) > 1:
        print(f' - Composites differ in length {sorted(lengths)}. Skipping.')
        continue

    make_figure(strain, kind, curves)

print('Analysis complete.')
