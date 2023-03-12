import numpy as np

from scipy.interpolate import RectBivariateSpline

def make_rotated_meshgrid(xspan, yspan, theta=0):
    '''
    Generate a meshgrid and rotate it clockwise by theta.

    Adapted from: https://stackoverflow.com/a/29709641

    Parameters
    ----------
    xspan : np.array, float
        x-coords of the unrotated grid.
    yspan : np.array, float
        y-coords of the unrotated grid.
    theta : float
        Angle, in radians, clockwise from the x-axis.

    Returns
    -------
    2D np.array, 2D np.array
        X, Y. xy-coords for each point in the generated meshgrid.

    '''
    xx, yy = np.meshgrid(xspan, yspan)

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])

    return np.einsum('ji, mni -> jmn', rotation_matrix, np.dstack([xx, yy]))

def make_mesh(center, theta, length, width, res_factor=1):
    '''
    Generate a meshgrid with the specified parameters.

    Parameters
    ----------
    center : 2-tuple, float
        Coordinates of the center of the grid.
    theta : float
        Angle of the line, in degrees, clockwise from the x-axis.
    length : int
        Length, in pixels.
    width : int
        Width, in pixels.
    res_factor : int
        Resolution factor. Number of interpolated points per pixel.

    Returns
    -------
    2D np.array, 2D np.array
        X, Y. xy-coords for each point in the generated meshgrid.

    '''
    cx, cy = center
    theta = np.radians(theta)

    # Number of points to interpolate:
    x_pts = int(np.round(res_factor * length))
    y_pts = int(np.round(res_factor * width))

    # Handle the edge case of a 1D grid.
    #
    # If width (or length) is set to 1px,
    # linspace() will give a single point:
    # -W/2 (or -L/2), and the 1D grid will
    # be offset from the midline.
    if x_pts > 1:
        x = np.linspace(-length/2, length/2, x_pts)
    else:
        x = np.zeros(2, dtype=float)

    if y_pts > 1:
        y = np.linspace(-width/2, width/2, y_pts)
    else:
        y = np.zeros(2, dtype=float)

    # Generate a grid centered at the origin.
    X, Y = make_rotated_meshgrid(x, y, theta)

    # Shift it to the specified midpoint.
    return X + cx, Y + cy

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
