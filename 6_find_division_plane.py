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

from scipy.optimize import least_squares
from scipy.signal.windows import gaussian

from util import load_stack
from util import file_exists
from util import write_textfile

from kymo import make_line
from kymo import find_profile

from fwhm import find_FWHM_intercepts
from fwhm import calc_FWHM
from fwhm import calc_FWHM_silent
from fwhm import FWHMError

KERNEL_SIZE = np.inf  # Should be about 1/4 the feature size.
KYMO_WIDTH = 8  # Length of kymograph line, in pixels.
KYMO_RESOLUTION = 8  # Number of points to interpolate per pixel.
MIN_FWHM_RATIO = 1.2

DEBUG = False

'''
First round of ring position optimization:
'''
def find_centroid(data):
    '''
    Find the centroid of a 2D distribution.

    Adapted from:
    https://scipy-cookbook.readthedocs.io/items/FittingData.html
    '''
    Y, X = np.indices(data.shape)

    cx = (X * data).sum() / data.sum()
    cy = (Y * data).sum() / data.sum()

    return cx, cy

'''
Second round of ring position optimization:
'''
def calc_FWHM_along_line(data, cx, cy, theta, length, resolution):

    profile = find_profile(data, cx, cy, theta, length, resolution)

    profile -= profile.min()

    fwhm = calc_FWHM(profile)

    if DEBUG:
        print(f'theta: {theta:.02f} -- fwhm: {fwhm:.02f} -- max: {profile.max():.0f}')

    return fwhm

def maximize_FWHM_vs_angle(data, cx, cy, theta0):
    '''
    For a fixed center point (cx, cy), find an angle theta that maximizes FWHM.
    '''
    p0 = [theta0]

    max_val = KYMO_RESOLUTION * KYMO_WIDTH
    obj_func = lambda p: max_val - calc_FWHM_along_line(data, cx, cy, *p, KYMO_WIDTH, KYMO_RESOLUTION)

    result = least_squares(obj_func, p0)

    return result

'''
Third round of ring position optimization:
'''
def integrate_along_line(data, cx, cy, theta, length, resolution, window=None):

    profile = find_profile(data, cx, cy, theta, length, resolution)

    if window is not None:
        profile = profile * window

    return profile.sum()

def maximize_signal_vs_position(data, cx0, cy0, theta):
    '''
    For a fixed angle theta, find a center point (cx, cy) that maximizes FWHM.
    '''
    width = KYMO_RESOLUTION * KYMO_WIDTH
    stdev = KYMO_RESOLUTION * KERNEL_SIZE
    window = gaussian(width, std=stdev)

    p0 = [cx0, cy0]

    max_val = data.sum()
    obj_func = lambda p: max_val - integrate_along_line(data, *p, theta, KYMO_WIDTH, KYMO_RESOLUTION, window=window)

    result = least_squares(obj_func, [cx0, cy0])

    return result

'''
Main loop:
'''
def find_ring_position(tif_file):

    basename = tif_file[:-len('.tif')]
    tsv_file = f'{basename}.ring_position.tsv'
    png_file = f'{basename}.ring_position.png'
    out_file = f'{basename}.ring_position.out'

    if file_exists(out_file):
        print(' - Already processed. Skipping.')
        return

    im = load_stack(tif_file)
    im_sum = im.sum(axis=0)

    '''
    1. Estimate ring position.
    '''
    cx1, cy1 = find_centroid(im_sum)

    '''
    2. Estimate ring orientation.
    '''
    try:
        result = maximize_FWHM_vs_angle(im_sum, cx1, cy1, 0)
    except ValueError as ex:
        print(' - Rejected. Optimizer aborted.')
        print(f' - Reason: ValueError: "{ex}"')
        write_textfile(out_file, 'Rejected.')
        return
    except FWHMError as ex:
        print(' - Rejected. FWHM calculation failed.')
        print(f' - Reason: FWHMError: "{ex}"')
        write_textfile(out_file, 'Rejected.')
        return

    if not result.success:
        print(' - Rejected. Optimizer failed.')
        print(f' - Reason: Solver: "{result.message}"')
        write_textfile(out_file, 'Rejected.')
        return

    theta1, = result.x

    # Wrap angle.
    theta1 %= 180

    '''
    3. Refine ring position.
    '''
    try:
        result = maximize_signal_vs_position(im_sum, cx1, cy1, theta1)
    except Exception as ex:
        print(' - Rejected. Center position refinement failed.')
        print(f' - Reason: "{ex}"')
        return

    if not result.success:
        print(' - Rejected. Optimizer failed.')
        print(f' - Reason: Solver: "{result.message}"')
        write_textfile(out_file, 'Rejected.')
        return

    cx2, cy2 = result.x

    '''
    Filter out low quality estimates.
    '''
    rejected = False

    profile_0deg = find_profile(im_sum, cx2, cy2, theta1, KYMO_WIDTH, KYMO_RESOLUTION)
    profile_90deg = find_profile(im_sum, cx2, cy2, theta1 + 90, KYMO_WIDTH, KYMO_RESOLUTION)

    fwhm_0deg = calc_FWHM_silent(profile_0deg)
    fwhm_90deg = calc_FWHM_silent(profile_90deg)

    if profile_0deg.min() > profile_90deg.min():
        print(' - Rejected. Shoulders too high.')
        print(f' - Reason: min(0deg) > min(90deg) -- {profile_0deg.min():.2f} > {profile_90deg.min():.2f}')
        write_textfile(out_file, 'Rejected.')
        rejected = True

    if (fwhm_0deg / fwhm_90deg < MIN_FWHM_RATIO) or np.isnan(fwhm_0deg) or np.isnan(fwhm_90deg):
        print(' - Rejected. FWHM ratio indicates poor fit.')
        print(f' - Reason: {fwhm_0deg / fwhm_90deg:.2f} < {MIN_FWHM_RATIO}')
        write_textfile(out_file, 'Rejected.')
        rejected = True

    if not rejected:
        print(' - Success.')

        # Save to disk.
        df = pd.DataFrame({
            'cx': [cx2],
            'cy': [cy2],
            'theta': [theta1],
            'FWHM_0deg': [fwhm_0deg],
            'FWHM_90deg': [fwhm_90deg],
        })
        df.to_csv(tsv_file, sep='\t', index=None)
        write_textfile(out_file, 'Success.')

    '''
    Generate plots for QC.
    '''
    plt.close('all')

    fig, axes = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    #
    # Image with estimated division plane
    #
    ax = axes[0]

    ax.imshow(im_sum, cmap='Greys_r')

    X, Y = make_line((cx1, cy1), theta1, KYMO_WIDTH, KYMO_RESOLUTION)
    ax.scatter(cx1, cy1, color='r', marker='s')
    ax.plot(X, Y, label='v1')

    X, Y = make_line((cx2, cy2), theta1, KYMO_WIDTH, KYMO_RESOLUTION)
    ax.scatter(cx2, cy2, color='g', marker='s')
    ax.plot(X, Y, label='v2')

    if rejected:
        msg = 'REJECTED'
        ax.text(.50, .01, msg,
                color='white',
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes)

    ax.legend(loc='upper right')

    #
    # Projection along and orthogonal to the division plane
    #
    ax = axes[1]

    ax.plot(profile_0deg,
            color='tab:orange',
            linestyle='solid',
            label=f'0˚ [{np.round(fwhm_0deg):.0f}]')

    if not np.isnan(fwhm_0deg):
        r1, r2 = find_FWHM_intercepts(profile_0deg)
        half_max = profile_0deg.max() / 2
        ax.hlines(half_max, r1, r2, color='red', linestyle=':')

    ax.plot(profile_90deg,
            color='tab:green',
            linestyle='dashed',
            label=f'90˚ [{np.round(fwhm_90deg):.0f}]')

    if not np.isnan(fwhm_90deg):
        r1, r2 = find_FWHM_intercepts(profile_90deg)
        half_max = profile_90deg.max() / 2
        ax.hlines(half_max, r1, r2, color='black', linestyle=':')

    ax.set_ylim(0, None)

    if rejected:
        msg = 'REJECTED'
        ax.text(.50, .01, msg,
                color='black',
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes)

    ax.legend(loc='upper right')

    fig.tight_layout()

    if rejected:
        png_file = f'{basename}.ring_position.rejected.png'
    fig.savefig(png_file)

def load_track_info(xml_file):

    basename = xml_file[:-len('.xml')]
    tracks_tsv = f'{basename}.TrackMetadata.tsv'

    df = pd.read_table(tracks_tsv)

    df['tif_file'] = basename + '.' + df.Track_Name + '.tif'

    # Skip tracks that were not registered.
    df = df[df.tif_file.apply(file_exists)]
    df = df.reset_index(drop=True)

    return df

def find_ring_positions(xml_file):

    df = load_track_info(xml_file)

    for n, row in df.iterrows():
        print(f'[{n+1}/{len(df)}] {row.tif_file}')
        find_ring_position(row.tif_file)

for n, xml_file in enumerate(xml_files):
    print(f'[{n+1}/{len(xml_files)}] {xml_file}')
    find_ring_positions(xml_file)

print('Analysis complete.')
