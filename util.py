import numpy as np
import os

from aicsimageio.readers import ome_tiff_reader
from aicsimageio.writers import ome_tiff_writer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_stack(filename):
    return ome_tiff_reader.TiffReader(filename, dim_order='TYX').data

def save_stack(filename, data):
    ome_tiff_writer.OmeTiffWriter.save(data, filename, dim_order='TYX')

def load_image(filename):
    return ome_tiff_reader.TiffReader(filename, dim_order='YX').data

def save_image(filename, data):
    ome_tiff_writer.OmeTiffWriter.save(data, filename, dim_order='YX')

def file_exists(filename):
    return os.path.isfile(filename)

def write_textfile(filename, msg):
    with open(filename, 'w') as fid:
        print(msg, file=fid)

def moving_average(x, win):
    return np.convolve(x, np.ones(win), 'valid') / win

def smooth(signal, smoothing=1):

    out = np.zeros_like(signal)
    out[smoothing-1:] = moving_average(signal, smoothing)

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
