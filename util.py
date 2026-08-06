import numpy as np
import os

from bioio import BioImage
from bioio_tifffile import Reader as TiffReader
from bioio_ome_tiff.writers import OmeTiffWriter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_stack(filename):
    return BioImage(filename).get_image_data('TYX')

def save_stack(filename, data):
    OmeTiffWriter.save(data, filename, dim_order='TYX')

def load_image(filename):
    # Raw MetaMorph frames are plain TIFF, not OME-TIFF.
    return BioImage(filename, reader=TiffReader).get_image_data('YX')

def save_image(filename, data):
    OmeTiffWriter.save(data, filename, dim_order='YX')

def file_exists(filename):
    return os.path.isfile(filename)

def dir_exists(filename):
    return os.path.isdir(filename)

def write_textfile(filename, msg):
    with open(filename, 'w') as fid:
        print(msg, file=fid)

def smooth(signal, win=1):
    '''
    Smooth via moving average with the specified window size.
    Pad the signal with the first and last smoothed values.
    '''
    out = np.convolve(signal, np.ones(win), 'same') / win

    out[:win//2] = out[win//2]
    out[-win//2:] = out[-win//2]

    return out

def stretch(signal):
    signal = np.array(signal)

    signal[np.abs(signal) == np.inf] = np.nan

    signal -= np.nanmin(signal)
    signal /= np.nanmax(signal)
    return signal

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
