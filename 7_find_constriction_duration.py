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
import traceback

from scipy.interpolate import UnivariateSpline

from util import load_stack
from util import file_exists
from util import smooth
from util import stretch
from util import find_longest_run

from kymo import make_line_endpoints
from kymo import make_kymograph

from fwhm import find_kymograph_peaks

SMOOTHING = 5  # rolling average window size
KYMO_WIDTH = 16  # kymograph width on orig image, in pixels
KYMO_RESOLUTION = 100  # kymograph interpolation width, in pixels
MIN_CONSTRICTION_TIME = 15  # min ring constriction duration
MIN_WIDTH_REBOUND = 5  # min width increase after constriction, out of KYMO_RESOLUTION

DEBUG = False

class ConstrictionError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

def find_constriction_start_and_end(fwhm_width, fwhm_area, peak_height):

    #
    # Find the longest region of decreasing ring width.
    #
    y = smooth(fwhm_width, SMOOTHING)
    x = np.arange(len(fwhm_width))

    spline = UnivariateSpline(x, y)
    deriv = spline.derivative()

    neg_slope = deriv(x) < 0
    t_start, run_len = find_longest_run(neg_slope)
    t_end = t_start + run_len

    #
    # Apply sanity checks.
    #
    if t_start <= 0:
        raise ConstrictionError('Truncated start of ring constriction.')

    if t_end >= len(fwhm_width) - 1:
        raise ConstrictionError('Truncated end of ring constriction.')

    if t_end - t_start < MIN_CONSTRICTION_TIME:
        raise ConstrictionError(f'Ring constriction duration too short. ({t_end - t_start})')

    max_rebound = np.diff(fwhm_width[t_end:]).cumsum().max()
    if max_rebound < MIN_WIDTH_REBOUND:
        raise ConstrictionError(f'Not enough width rebound after end of ring constriction. ({max_rebound:.01f} < {MIN_WIDTH_REBOUND})')

    if np.argmax(fwhm_area) > t_end:
        raise ConstrictionError('Ring constriction ends before total ring intensity peaks.')

    if np.argmax(peak_height) > t_end:
        raise ConstrictionError('Ring constriction ends before maximum ring intensity peaks.')

    return t_start, t_end

def find_ring_timing(tif_file, track_start):

    basename = tif_file[:-len('.tif')]
    div_file = f'{basename}.ring_position.tsv'
    out_file = f'{basename}.ring_timing.tsv'
    npz_file = f'{basename}.ring_timing.npz'
    png_file = f'{basename}.ring_timing.png'

    if file_exists(out_file):
        print(' - Division parameters file already exists. Skipping.')
        return

    if not file_exists(div_file):
        print(' - Coordinates file missing. Skipping.')
        return

    #
    # Load tiff stack and ring parameters.
    #
    im = load_stack(tif_file)

    df = pd.read_table(div_file)
    cx, cy, theta = df.iloc[0][['cx', 'cy', 'theta']]

    #
    # Generate a kymograph.
    #
    (x1, y1), (x2, y2) = make_line_endpoints((cx, cy), theta, KYMO_WIDTH)
    line = x1, x2, y1, y2

    kymograph = make_kymograph(im, line, KYMO_RESOLUTION)

    #
    # Find kymograph peaks via FWMH.
    #
    try:
        result = find_kymograph_peaks(kymograph)
    except Exception as ex:
        print(' - Failed.')
        print(f' - Reason: {ex}')
        with open(f'{basename}.ring_timing.error', 'w') as fid:
            print(ex, file=fid)
            print(file=fid)
            print(traceback.format_exc(), file=fid)
        return

    fwhm_loc, fwhm_width, fwhm_area, peak_loc, peak_height, r1, r2 = result

    np.savez(npz_file,
             fwhm_loc=fwhm_loc,
             fwhm_width=fwhm_width,
             fwhm_area=fwhm_area,
             peak_loc=peak_loc,
             peak_height=peak_height,
             r1=r1,
             r2=r2)

    #
    # Extract parameters.
    #
    try:
        t_start, t_end = find_constriction_start_and_end(fwhm_width, fwhm_area, peak_height)
    except Exception as ex:
        print(' - Failed.')
        print(f' - Reason: {ex}')
        with open(f'{basename}.ring_timing.error', 'w') as fid:
            print(ex, file=fid)
            print(file=fid)
            print(traceback.format_exc(), file=fid)
        return

    print(' - Success.')

    # Save to disk.
    df = pd.DataFrame({
        't_start': [t_start],
        't_end': [t_end],
        'track_start': [track_start],
    })
    df.to_csv(out_file, sep='\t', index=None)

    '''
    Generate QC plots.
    '''
    plt.close('all')

    #
    # Initialize plots
    #
    fig, axes = plt.subplots(4, 5)
    fig.set_figheight(5 * axes.shape[0])
    fig.set_figwidth(5 * axes.shape[1])

    #
    # Composite image used to find division plane
    #
    ax = axes[0, 0]
    ax.imshow(im.sum(axis=0))
    ax.scatter(cx, cy, color='r', marker='s')
    ax.plot((x1, x2), (y1, y2))

    #
    # Kymograph along division plane
    #
    ax = axes[0, 1]
    ax.imshow(kymograph)
    w = kymograph.shape[1]
    x = range(w)
    y = [t_start] * w
    ax.plot(x, y, color='g', linestyle=':')
    y = [t_end] * w
    ax.plot(x, y, color='r', linestyle=':')

    #
    # FWHM peak location and width
    #
    ax = axes[0, 2]
    ax.plot(smooth(fwhm_loc, SMOOTHING), label='loc')
    ax.plot(smooth(fwhm_width, SMOOTHING), label='width')
    _, ymax = 0, 100
    ax.vlines(t_start, 0, ymax, colors='g', linestyles=':', label=f't_start: {t_start}')
    ax.vlines(t_end, 0, ymax, colors='r', linestyles=':', label=f't_end: {t_end}')
    ax.set_ylim(0, ymax)
    ax.legend(loc='lower left')

    #
    # FWHM peak height
    #
    ax = axes[0, 3]
    ax.plot(smooth(peak_height, SMOOTHING), label='height')
    _, ymax = ax.get_ylim()
    ax.vlines(t_start, 0, ymax, colors='g', linestyles=':', label=f't_start: {t_start}')
    ax.vlines(t_end, 0, ymax, colors='r', linestyles=':', label=f't_end: {t_end}')
    ax.set_ylim(0, None)
    ax.legend(loc='lower left')

    #
    # Stretched FWHM peak width, height, and area
    #
    ax = axes[0, 4]
    ax.plot(smooth(stretch(fwhm_width), SMOOTHING), label='width')
    ax.plot(smooth(stretch(peak_height), SMOOTHING), label='height')
    ax.plot(smooth(stretch(fwhm_area), SMOOTHING), label='area')
    ax.vlines(t_start, 0, 1, colors='g', linestyles=':', label=f't_start: {t_start}')
    ax.vlines(t_end, 0, 1, colors='r', linestyles=':', label=f't_end: {t_end}')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower left')

    #
    # Division +/-7 frames
    #
    show_frame(axes[1], im, t_end - 7)
    show_frame(axes[2], im, t_end - 2)
    show_frame(axes[3], im, t_end + 3)

    ax = axes[2, 2]
    ax.text(.50, .01, 'END OF CONSTRICTION',
            color='white',
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax.transAxes)

    #
    # Save to disk.
    #
    fig.tight_layout()
    fig.savefig(png_file)
    
def show_frame(axes, im, start_frame):

    kwargs = {
        'cmap': 'Greys_r',
        'vmin': im.min(),
        'vmax': im.max(),
    }

    for i, ax in enumerate(axes):

        frame = start_frame + i

        if frame < im.shape[0]:
            ax.imshow(im[frame], **kwargs)
            ax.text(.01, .99, f'FRAME {frame}',
                    color='red',
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)
        else:
            ax.remove()

def load_track_info(xml_file):

    basename = xml_file[:-len('.xml')]
    tracks_tsv = f'{basename}.TrackMetadata.tsv'

    df = pd.read_table(tracks_tsv)
    df = df[['Track_Name', 'TRACK_START']]

    df['TRACK_START'] = df['TRACK_START'].astype(int)
    df['tif_file'] = basename + '.' + df.Track_Name + '.tif'
    df['ring_pos_tsv'] = basename + '.' + df.Track_Name +  '.ring_position.tsv'

    # Skip tracks where finding the ring position failed.
    df = df[df.ring_pos_tsv.apply(file_exists)]
    df = df.reset_index(drop=True)

    return df

def find_ring_timings(xml_file):

    df = load_track_info(xml_file)

    for n, row in df.iterrows():
        print(f'[{n+1}/{len(df)}] {row.tif_file}')
        find_ring_timing(row.tif_file, row.TRACK_START)

for n, xml_file in enumerate(xml_files):
    print(f'[{n+1}/{len(xml_files)}] {xml_file}')
    find_ring_timings(xml_file)

print('Analysis complete.')
