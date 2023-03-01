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
from scipy.optimize import least_squares

# MIN_ECCENTRICITY = .5  # 0 = circle, 1 = hyperbola
MAX_SIGMA = 4  # High sigma means we prob aligned along cell axis.

def load_image(filename):
    return tiff_reader.TiffReader(filename, dim_order='TYX').data

def save_results(filename, cx, cy, sigma_x, sigma_y, theta, ecc):
    df = pd.DataFrame(
        data = [(cx, cy, sigma_x, sigma_y, theta, ecc)],
        columns = ['cx', 'cy', 'sigma_x', 'sigma_y', 'theta', 'ecc']
    )
    df.to_csv(filename, sep='\t', index=None)

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

def calc_moments(data):
    '''
    Estimate the Gaussian parameters of a 2D distribution by calculating its moments.
    '''
    # Find the centroid.
    X, Y = np.indices(data.shape)
    
    cx = (X * data).sum() / data.sum()
    cy = (Y * data).sum() / data.sum()

    # Find the width around the centroid.
    row = data[int(cx), :]
    col = data[:, int(cy)]

    width_x = np.sqrt(np.abs((np.arange(col.size) - cx) ** 2 * col).sum() / col.sum())
    width_y = np.sqrt(np.abs((np.arange(row.size) - cy) ** 2 * row).sum() / row.sum())
    
    # Find the height.
    height = data.max()
    
    return height, cx, cy, width_x, width_y

def Gaussian2D(x, y, A, x0, y0, sigma_x, sigma_y, theta):
    '''
    Calculate the values of a 2D gaussian at (x, y) with the given parameters.
    
    theta is the angle of the semimajor axis, in degrees, measured clockwise from the x-axis.
    '''
    theta = np.radians(theta)
    sigx2 = sigma_x ** 2
    sigy2 = sigma_y ** 2

    a = np.cos(theta) ** 2 / (2 * sigx2) + np.sin(theta) ** 2 / (2 * sigy2)
    b = np.sin(theta) ** 2 / (2 * sigx2) + np.cos(theta) ** 2 / (2 * sigy2)
    c = - np.sin(2 * theta) / (4 * sigx2) + np.sin(2 * theta) / (4 * sigy2)
    
    expo = a * (x - x0) ** 2 + b * (y - y0) ** 2 + 2 * c * (x - x0) * (y - y0)

    return A * np.exp(-expo)

def fit_Gaussian2D(data):
    '''
    Fit a 2D Gaussian to the data.

    Returns: (A, y0, x0, sigma_y, sigma_x, theta)
    
    theta is the angle of the semimajor axis, in degrees, measured clockwise from the x-axis.
    '''
    # xy: coordinates to evaluate the Gaussian at.
    # p0: initial guess of the Gaussian parameters.
    #   - Find the moments of the data.
    #   - Tack on an initial guess of 0 degrees for theta.
    xy = np.indices(data.shape)
    p0 = *calc_moments(data), 0

    # Construct an objective function.
    #   - input: six Gaussian parameters
    #   - output: a 1D array of pixelwise errors
    objective_function = lambda p: np.ravel(Gaussian2D(*xy, *p) - data)

    # Run the optimizer.
    result = least_squares(objective_function, p0)

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
        result = fit_Gaussian2D(im_sum)
    except ValueError as ex:
        print(' - Rejected. 2D Gaussian fit failed.')
        print(f' - Reason: ValueError: "{ex}"')
        return

    if not result.success:
        print(' - Rejected. 2D Gaussian fit failed.')
        print(f' - Reason: Solver: "{result.message}"')
        return

    A, cy, cx, sigma_y, sigma_x, theta = result.x

    if sigma_x > sigma_y:
        a, b = sigma_x, sigma_y
    else:
        a, b = sigma_y, sigma_x
        theta += 90
    # eccentricity = np.sqrt(1. - (b / a) ** 2)
    # 
    # if eccentricity < MIN_ECCENTRICITY:
    #     print(' - Rejected. Gaussian not eccentric enough.')
    #     print(f' - Reason: ecc = {eccentricity:.02f} < {MIN_ECCENTRICITY}')
    #     return

    largest_sigma = max(sigma_x, sigma_y)
    if largest_sigma > MAX_SIGMA:
        print(' - Rejected. Gaussian sigma too large. Possibly aligned along cell body.')
        print(f' - Reason: sigma = {largest_sigma:.02f} > {MAX_SIGMA}')
        return

    # Wrap angle.
    theta %= 180

    print(' - Success.')

    # Save to disk.
    df = pd.DataFrame({
        'cx': [cx],
        'cy': [cy],
        'theta': [theta],
    })
    df.to_csv(out_file, sep='\t', index=None)

    # Save debugging plot.
    (x1, y1), (x2, y2) = make_line_endpoints((cx, cy), theta, 12)

    plt.close('all')
    plt.imshow(im_sum)
    plt.scatter(cx, cy, color='r', marker='s')
    plt.plot((x1, x2), (y1, y2))
    plt.savefig(png_file)

for n, filename in enumerate(tif_files):
    print(f'[{n+1}/{len(tif_files)}] {filename}')
    find_division_plane(filename)

print('Analysis complete.')
