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

from aicsimageio.readers import tiff_reader
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
from scipy.signal import gaussian

# from scipy.ndimage import gaussian_filter
# BLUR_RADIUS = .5
# data = gaussian_filter(data, BLUR_RADIUS)

KERNEL_SIZE = 3  # Should be about 1/4 the feature size.
KYMO_WIDTH = 12
KYMO_RESOLUTION = 100
MIN_FWHM_RATIO = 1.5

debug = False

def load_image(filename):
    return tiff_reader.TiffReader(filename, dim_order='TYX').data

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

def interp_along_line(im, line, resolution):

    ny, nx = im.shape

    y = np.arange(ny)
    x = np.arange(nx)

    sp = RectBivariateSpline(y, x, im)

    x1, x2, y1, y2 = line

    y = np.linspace(y1, y2, resolution)
    x = np.linspace(x1, x2, resolution)

    return sp.ev(y, x)

def find_profile(data, cx, cy, theta, length, resolution):

    (x1, y1), (x2, y2) = make_line_endpoints((cx, cy), theta, length)
    line = (x1, x2, y1, y2)

    profile = interp_along_line(data, line, resolution)

    return profile

def gaussian_kernel(width, stdev):
    '''
    Generate a 2D Gaussian kernel.
    '''
    kernel_1D = gaussian(width, std=stdev)
    kernel_2D = np.outer(kernel_1D, kernel_1D)
    return kernel_2D

def find_centroid(data):
    '''
    Find the centroid of a 2D distribution.

    Adapted from:
    https://scipy-cookbook.readthedocs.io/items/FittingData.html
    '''
    Y, X = np.indices(data.shape)

    cy = (Y * data).sum() / data.sum()
    cx = (X * data).sum() / data.sum()

    return cy, cx

def find_roots_around_peak(roots, peak_loc):
    
    s = roots - peak_loc
    i = np.where(np.sign(s[:1]) != np.sign(s[1:]))[0][0]

    r1 = roots[i]
    r2 = roots[i+1]
    
    return r1, r2

class FWHMError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

def find_FWHM_intercepts(signal):

    peak_loc = np.argmax(signal)
    half_max = np.max(signal) / 2
    
    x = np.arange(len(signal))
    y = signal - half_max

    spline = UnivariateSpline(x, y, s=0)
    roots = spline.roots()

    if len(roots) == 2:
        r1, r2 = roots
    elif len(roots) > 2:
        if peak_loc < roots.min() or peak_loc > roots.max():
            raise FWHMError('Max value does not occur between roots.')

        print(f'Using root disambiguation: {peak_loc} in {roots}')
        r1, r2 = find_roots_around_peak(roots, peak_loc)
    else:
        raise FWHMError('Did not find at least 2 roots in FWHM calculation.')

    return r1, r2

def calc_FWHM(signal):

    r1, r2 = find_FWHM_intercepts(signal)

    return r2 - r1

def calc_FWHM_along_line(data, cx, cy, theta, length, resolution):

    profile = find_profile(data, cx, cy, theta, length, resolution)

    profile -= profile.min()

    fwhm = calc_FWHM(profile)

    if debug:
        print(f'theta: {theta:.02f} -- fwhm: {fwhm:.02f} -- max: {profile.max():.0f}')

    return fwhm

def maximize_FWHM(data):

    data_middle = data * gaussian_kernel(len(data), KERNEL_SIZE)
    cx, cy = find_centroid(data)

    obj_func = lambda p: KYMO_RESOLUTION - calc_FWHM_along_line(data, cx, cy, *p, KYMO_WIDTH, KYMO_RESOLUTION)

    result = least_squares(obj_func, 0)

    return result

def find_division_plane(filename):

    basename = filename[:-len('.tif')]
    out_file = f'{basename}.division_plane.tsv'
    png_file = f'{basename}.division_plane.png'

    if file_exists(out_file):
        print(' - Coordinates file already exists. Skipping.')
        return

    im = load_image(filename)
    im_sum = im.sum(axis=0)

    try:
        result = maximize_FWHM(im_sum)
    except ValueError as ex:
        print(' - Rejected. Optimizer aborted.')
        print(f' - Reason: ValueError: "{ex}"')
        return
    except FWHMError as ex:
        print(' - Rejected. FWHM calculation failed.')
        print(f' - Reason: FWHMError: "{ex}"')
        return

    if not result.success:
        print(' - Rejected. Optimizer failed.')
        print(f' - Reason: Solver: "{result.message}"')
        return

    theta = result.x[0]

    '''
    Check result quality.
    '''
    rejected = False

    im_sum_middle = im_sum * gaussian_kernel(len(im_sum), KERNEL_SIZE)
    cy, cx = find_centroid(im_sum_middle)
    profile1 = find_profile(im_sum, cx, cy, theta, KYMO_WIDTH, KYMO_RESOLUTION)
    profile2 = find_profile(im_sum, cx, cy, theta + 90, KYMO_WIDTH, KYMO_RESOLUTION)

    try:
        fwhm1 = calc_FWHM(profile1)
    except FWHMError:
        fwhm1 = np.nan
        rejected = True
    try:
        fwhm2 = calc_FWHM(profile2)
    except FWHMError:
        fwhm2 = np.nan
        rejected = True

    if (fwhm1 / fwhm2 < MIN_FWHM_RATIO) or np.isnan(fwhm1) or np.isnan(fwhm2):
        print(' - Rejected. FWHM ratio indicates poor fit.')
        print(f' - Reason: {fwhm1 / fwhm2:.2f} < MIN_FWHM_RATIO')
        rejected = True

    # Wrap angle.
    theta %= 180

    if not rejected:
        print(' - Success.')

        # Save to disk.
        df = pd.DataFrame({
            'cx': [cx],
            'cy': [cy],
            'theta': [theta],
            'fwhm1': [fwhm1],
            'fwhm2': [fwhm2],
        })
        df.to_csv(out_file, sep='\t', index=None)

    #
    # Save debugging plot.
    #
    (x1, y1), (x2, y2) = make_line_endpoints((cx, cy), theta, KYMO_WIDTH)

    plt.close('all')
    plt.imshow(im_sum, cmap='Greys_r')
    plt.scatter(cx, cy, color='r', marker='s')
    plt.plot((x1, x2), (y1, y2))

    msg = ''
    if rejected:
        msg += 'REJECTED\n'
        png_file = f'{basename}.division_plane.rejected.png'
    msg += f'fwhm1 = {fwhm1:.02f}\n'
    msg += f'fwhm2 = {fwhm2:.02f}'
    ax = plt.gca()
    ax.text(.99, .01, msg,
            color='white',
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes)

    plt.savefig(png_file)

for n, filename in enumerate(tif_files):
    print(f'[{n+1}/{len(tif_files)}] {filename}')
    find_division_plane(filename)

print('Analysis complete.')
