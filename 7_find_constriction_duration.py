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
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit

'''
If curve_fit() is throwing these warnings:
 * OptimizeWarning: Covariance of the parameters could not be estimated
try increasing KYMO_WIDTH.
'''
SMOOTHING = 8  # rolling average window size
KYMO_WIDTH = 16  # kymograph width on orig image, in pixels
KYMO_RESOLUTION = 100  # kymograph interpolation width, in pixels

def load_image(filename):
    return tiff_reader.TiffReader(filename, dim_order='TYX').data

def file_exists(filename):
    return pathlib.Path(filename).exists()

def moving_average(x, win):
    return np.convolve(x, np.ones(win), 'valid') / win

def stretch(signal):
    signal = np.array(signal)

    signal[np.abs(signal) == np.inf] = np.nan

    signal -= signal.min()
    signal /= signal.max()
    return signal

def smooth(signal):

    out = np.zeros(len(signal))
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

def Gaussian(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + offset

def fit_Gaussian(Y, X=None):

    if X is None:
        X = np.arange(len(Y))

    # Initial guess:
    a = Y.max() - Y.min()
    m = X.mean()
    s = X.std()
    o = Y.min()

    popt, pcov = curve_fit(Gaussian, X, Y, p0=[a, m, s, o])
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

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

    im = load_image(filename)

    df = pd.read_table(div_file)
    cx, cy, theta = df.iloc[0][['cx', 'cy', 'theta']]

    (x1, y1), (x2, y2) = make_line_endpoints((cx, cy), theta, KYMO_WIDTH)
    line = x1, x2, y1, y2

    kymograph = make_kymograph(im, line, KYMO_RESOLUTION)

    gaussians = [fit_Gaussian(row) for row in kymograph]
    A_fit, mu_fit, sigma_fit, offset_fit = np.array(gaussians)[:,0,:].T

    objective = smooth(stretch(offset_fit)) - smooth(stretch(sigma_fit))
    t_end = np.argmax(objective)

    print(' - Success.')

    # Save to disk.
    df = pd.DataFrame({
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
    y = [t_end] * w
    ax.plot(x, y)

    #
    # Intensity profile along division plane
    #
    ax = axes[0, 2]
    ax.plot(kymograph.sum(axis=0))

    #
    # Less informative Gaussian fit parameters
    #
    ax = axes[0, 3]
    ax.plot(smooth(stretch(A_fit)), label='A')
    ax.plot(smooth(stretch(mu_fit)), label='mu')
    ax.vlines(t_end, 0, 1, colors='r', linestyles=':', label=f't_end: {t_end}')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower left')

    #
    # More informative Gaussian fit parameters
    #
    ax = axes[0, 4]
    ax.plot(smooth(stretch(sigma_fit)), label='sigma')
    ax.plot(smooth(stretch(offset_fit)), label='offset')
    ax.plot(objective, label='objective', color='k', linestyle=':', alpha=.5)
    ax.vlines(t_end, -1, 1, colors='r', linestyles=':', label=f't_end: {t_end}')
    ax.set_ylim(-1, 1)
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
            ax.text(.02, .01, f'FRAME {frame}',
                    color='red',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax.transAxes)
        else:
            ax.remove()

for n, filename in enumerate(tif_files):
    print(f'[{n+1}/{len(tif_files)}] {filename}')
    extract_division_parameters(filename)

print('Analysis complete.')
