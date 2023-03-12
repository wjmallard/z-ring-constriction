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

def make_line_endpoints(center, theta, length):
    '''
    Generate endpoint xy-coords of a line with the specified parameters.

    Parameters
    ----------
    center : 2-tuple, float
        Coordinates of the center of the line.
    theta : float
        Angle of the line, in degrees, clockwise from the x-axis.
    length : int
        Length, in pixels.

    Returns
    -------
    (float, float), (float, float)
        P1, P2. xy-coords for each end of the generated line.

    '''
    theta = np.radians(theta)

    X = np.array((-length/2, length/2))
    Y = np.zeros(2)

    X_rot = X * np.cos(theta)
    Y_rot = X * np.sin(theta)

    x1, x2 = center[0] + X_rot
    y1, y2 = center[1] + Y_rot

    return (x1, y1), (x2, y2)

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
