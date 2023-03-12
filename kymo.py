import numpy as np

from scipy.interpolate import RectBivariateSpline

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
