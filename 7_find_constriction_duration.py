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
import traceback

from aicsimageio.readers import tiff_reader
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RectBivariateSpline
from scipy.signal import find_peaks, peak_widths

'''
If curve_fit() is throwing these warnings:
 * OptimizeWarning: Covariance of the parameters could not be estimated
try increasing KYMO_WIDTH.
'''
SMOOTHING = 5  # rolling average window size
KYMO_WIDTH = 16  # kymograph width on orig image, in pixels
KYMO_RESOLUTION = 100  # kymograph interpolation width, in pixels
MIN_CONSTRICTION_TIME = 15  # min ring constriction duration
MIN_WIDTH_REBOUND = 5  # min width increase after constriction, out of KYMO_RESOLUTION

DEBUG = False

def load_image(filename):
    return tiff_reader.TiffReader(filename, dim_order='TYX').data

def file_exists(filename):
    return pathlib.Path(filename).exists()

def moving_average(x, win):
    return np.convolve(x, np.ones(win), 'valid') / win

def stretch(signal):
    signal = np.array(signal)

    signal[np.abs(signal) == np.inf] = np.nan

    signal -= np.nanmin(signal)
    signal /= np.nanmax(signal)
    return signal

def smooth(signal):

    out = np.zeros_like(signal)
    out[SMOOTHING-1:] = moving_average(signal, SMOOTHING)

    return out

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

def find_roots_around_peak(roots, peak_loc):

    s = roots - peak_loc
    i = np.where(np.sign(s[:1]) != np.sign(s[1:]))[0][0]

    r1 = roots[i]
    r2 = roots[i+1]

    return r1, r2

class FWHMError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

def find_primary_peak(signal, half_max=None):
    '''
    Find the primary peak in a signal via FWHM.

    Allow the user to pass in a global half-max value.
    If one is not provided, use half-max of the signal.

    Return: fwhm_loc, fwhm_width, fwhm_area, ...
    '''
    if half_max is None:
        half_max = np.max(signal) / 2

    if np.max(signal) < half_max:
        return [np.nan] * 7

    #
    # Find peaks taller than half-max. Sort them by height.
    #
    peak_locs, props = find_peaks(signal, height=half_max)

    peaks = list(zip(peak_locs, props['peak_heights']))
    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

    peak_loc, peak_height = peaks[0]

    #
    # Find all intercepts of the half-max line.
    #
    x = np.arange(len(signal))
    y = signal - half_max

    spline = UnivariateSpline(x, y)
    roots = spline.roots()

    if len(roots) == 2:
        r1, r2 = roots
    elif len(roots) > 2:
        if peak_loc < roots.min() or peak_loc > roots.max():
            raise FWHMError('Max value does not occur between roots.')

        if DEBUG:
            print(f'Using root disambiguation: {peak_loc} in {roots}')
        r1, r2 = find_roots_around_peak(roots, peak_loc)
    else:
        return [np.nan] * 7

    #
    # Find FWHM location, width, and area.
    #
    fwhm_loc = np.mean((r1, r2))
    fwhm_width = r2 - r1

    spline = UnivariateSpline(x, signal)
    fwhm_area = spline.integral(r1, r2)

    return fwhm_loc, fwhm_width, fwhm_area, peak_loc, peak_height, r1, r2

def find_runs(X):
    '''
    Find all True runs in a boolean array.
    '''
    X = np.array(X).astype(bool)

    false_locs = np.where(~X)[0]
    false_locs = np.hstack((-1, false_locs, len(X)))

    gap_lens = np.diff(false_locs)
    gap_ends = np.cumsum(gap_lens)

    run_lens = gap_lens - 1
    run_locs = np.hstack((0, gap_ends[:-1]))

    locs = run_locs[run_lens > 0]
    lens = run_lens[run_lens > 0]

    return tuple(zip(locs, lens))

def find_longest_run(X):
    '''
    Find the longest True run in a boolean array.
    '''
    runs = find_runs(X)

    key = lambda x: (x[1], x[0])
    runs = sorted(runs, key=key, reverse=True)

    return runs[0]

class ConstrictionError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

def find_constriction_start_and_end(fwhm_width, fwhm_area, peak_height):

    #
    # Find the longest region of decreasing ring width.
    #
    y = smooth(fwhm_width)
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

def find_start_of_stable_loc(fwhm_loc):
    pass

def extract_division_parameters(filename):

    basename = filename[:-len('.tif')]
    div_file = f'{basename}.division_plane.tsv'
    out_file = f'{basename}.division_params.tsv'
    png_file = f'{basename}.division_params.png'

    if file_exists(out_file):
        print(' - Division parameters file already exists. Skipping.')
        return

    if not file_exists(div_file):
        print(' - Coordinates file missing. Skipping.')
        return

    #
    # Load tiff stack and ring parameters.
    #
    im = load_image(filename)

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
        peaks = [find_primary_peak(row) for row in kymograph]
        fwhm_loc, fwhm_width, fwhm_area, peak_loc, peak_height, r1, r2 = np.array(peaks).T
    except Exception as ex:
        print(' - Failed.')
        print(f' - Reason: {ex}')
        with open(f'{basename}.division_params.error', 'w') as fid:
            print(ex, file=fid)
            print(file=fid)
            print(traceback.format_exc(), file=fid)
        return

    #
    # Extract parameters.
    #
    try:
        t_start, t_end = find_constriction_start_and_end(fwhm_width, fwhm_area, peak_height)
    except Exception as ex:
        print(' - Failed.')
        print(f' - Reason: {ex}')
        with open(f'{basename}.division_plane.error', 'w') as fid:
            print(ex, file=fid)
            print(file=fid)
            print(traceback.format_exc(), file=fid)
        return

    print(' - Success.')

    # Save to disk.
    df = pd.DataFrame({
        't_start': [t_start],
        't_end': [t_end],
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
    # Intensity profile along division plane
    #
    ax = axes[0, 2]
    #ax.plot(kymograph.sum(axis=0))
    ax.plot(smooth(fwhm_loc), label='loc')
    ax.plot(smooth(fwhm_width), label='width')
    _, ymax = 0, 100
    ax.vlines(t_start, 0, ymax, colors='g', linestyles=':', label=f't_start: {t_start}')
    ax.vlines(t_end, 0, ymax, colors='r', linestyles=':', label=f't_end: {t_end}')
    ax.set_ylim(0, ymax)
    ax.legend(loc='lower left')

    #
    # Less informative Gaussian fit parameters
    #
    ax = axes[0, 3]
    ax.plot(smooth(peak_height), label='height')
    _, ymax = ax.get_ylim()
    ax.vlines(t_start, 0, ymax, colors='g', linestyles=':', label=f't_start: {t_start}')
    ax.vlines(t_end, 0, ymax, colors='r', linestyles=':', label=f't_end: {t_end}')
    ax.set_ylim(0, None)
    ax.legend(loc='lower left')

    # x = np.arange(len(fwhm_loc))
    # y = fwhm_loc
    # spline = UnivariateSpline(x, y, k=1)
    #
    # ax.plot(fwhm_loc, label='loc')
    # for knot in spline.get_knots():
    #     ax.vlines(knot, 0, KYMO_RESOLUTION, color='red', linestyle=':', linewidth=.5)
    # ax.legend(loc='lower left')

    #
    # More informative Gaussian fit parameters
    #
    ax = axes[0, 4]
    ax.plot(smooth(stretch(fwhm_width)), label='width')
    ax.plot(smooth(stretch(peak_height)), label='height')
    ax.plot(smooth(stretch(fwhm_area)), label='area')
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

for n, filename in enumerate(tif_files):
    print(f'[{n+1}/{len(tif_files)}] {filename}')
    extract_division_parameters(filename)

print('Analysis complete.')
