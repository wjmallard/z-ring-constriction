import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

class FWHMError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

def find_kymograph_peaks(kymograph):
    '''
    Returns: 2D numpy array with the following rows:
     - fwhm_loc, fwhm_width, fwhm_area, peak_loc, peak_height, r1, r2
    '''
    peaks = [find_primary_peak(row) for row in kymograph]
    return np.array(peaks).T

def find_roots_around_peak(roots, peak_loc):
    
    s = roots - peak_loc
    i = np.where(np.sign(s[:1]) != np.sign(s[1:]))[0][0]

    r1 = roots[i]
    r2 = roots[i+1]
    
    return r1, r2

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

    #
    # If no peaks are found, return a list of NaNs.
    #
    if not peaks:
        return [np.nan] * 7

    #
    # Take the tallest peak.
    #
    peak_loc, peak_height = peaks[0]

    #
    # Find peak boundaries.
    #
    r1, r2 = find_FWHM_intercepts(signal, peak_loc, peak_height / 2)

    #
    # Find peak location, width, and area.
    #
    fwhm_loc = np.mean((r1, r2))
    fwhm_width = r2 - r1

    y = signal
    x = np.arange(len(y))

    spline = UnivariateSpline(x, y, s=0)
    fwhm_area = spline.integral(r1, r2)

    return fwhm_loc, fwhm_width, fwhm_area, peak_loc, peak_height, r1, r2

def find_FWHM_intercepts(signal, peak_loc=None, half_max=None):

    if peak_loc is None:
        peak_loc = np.argmax(signal)
    if half_max is None:
        half_max = np.max(signal) / 2

    if peak_loc < 0 or peak_loc >= len(signal):
        raise FWHMError('Received invalid peak_loc value.')
    if np.max(signal) < half_max:
        raise FWHMError('Received invalid half_max value.')

    #
    # Find all intercepts of the half-max line.
    #
    y = signal - half_max
    x = np.arange(len(y))

    spline = UnivariateSpline(x, y, s=0)
    roots = spline.roots()

    if len(roots) < 2:
        raise FWHMError('Did not find at least 2 roots in FWHM calculation.')

    #
    # Select the outer-most intercepts.
    #
    r1 = roots.min()
    r2 = roots.max()

    if peak_loc < r1 or peak_loc > r2:
        raise FWHMError('Max value does not occur between roots.')

    return r1, r2

def calc_FWHM(signal):

    r1, r2 = find_FWHM_intercepts(signal)

    return r2 - r1

def calc_FWHM_silent(signal):

    try:
        r1, r2 = find_FWHM_intercepts(signal)
    except FWHMError:
        return np.nan

    return r2 - r1

def find_xval(signal, yval):

    y = signal - yval
    x = np.arange(len(y))

    spline = UnivariateSpline(x, y, s=0)
    roots = spline.roots()

    if len(roots) > 1:
        print('Warning: Found multiple roots. Returning the first one.')

    return roots[0]
