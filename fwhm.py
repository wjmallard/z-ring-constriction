import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

DEBUG = False

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

        if DEBUG:
            print(f'Using root disambiguation: {peak_loc} in {roots}')
        r1, r2 = find_roots_around_peak(roots, peak_loc)
    else:
        raise FWHMError('Did not find at least 2 roots in FWHM calculation.')

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
