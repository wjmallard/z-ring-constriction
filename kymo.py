import numpy as np

from scipy.interpolate import RectBivariateSpline

def make_line(center, theta, length, resolution=1):

    cx, cy = center
    theta = np.radians(theta)
    num_pts = length * resolution

    x = np.linspace(-length/2, length/2, num_pts)

    X = x * np.cos(theta)
    Y = x * np.sin(theta)

    return X + cx, Y + cy

def make_lines(center, theta, length, resolution=1, width=1, w_res=1):

    centers = make_line(center, theta + 90, width, w_res)
    centers = np.array(centers).T

    lines = [make_line(c, theta, length, resolution) for c in centers]    
    lines = np.array(lines)

    return lines

def find_profile(im, cx, cy, theta, length, resolution):

    ny, nx = im.shape

    y = np.arange(ny)
    x = np.arange(nx)

    sp = RectBivariateSpline(y, x, im)

    X, Y = make_line((cx, cy), theta, length, resolution)

    return sp.ev(Y, X)

def make_kymograph(stack, cx, cy, theta, length, resolution):

    _, ny, nx = stack.shape

    y = np.arange(ny)
    x = np.arange(nx)

    splines = [RectBivariateSpline(y, x, frame) for frame in stack]

    X, Y = make_line((cx, cy), theta, length, resolution)

    interpolated_lines = [sp.ev(Y, X) for sp in splines]

    kymograph = np.array(interpolated_lines)

    return kymograph

def make_kymograph_grid(stack, cx, cy, theta, length, resolution, width, w_res=1):

    _, ny, nx = stack.shape

    y = np.arange(ny)
    x = np.arange(nx)

    splines = [RectBivariateSpline(y, x, frame) for frame in stack]

    XY = make_lines((cx, cy), theta, length, resolution, width, w_res)

    interpolated_lines = [[sp.ev(y, x) for sp in splines] for (x, y) in XY]

    kymograph = np.array(interpolated_lines)
    kymograph = kymograph.mean(axis=0)

    return kymograph
