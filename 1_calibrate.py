#!/usr/bin/env python
import sys
import os
try:
    calib_dir = sys.argv[1]
    assert os.path.isdir(calib_dir), 'Calibration directory does not exist.'
except:
    script = sys.argv[0].split('/')[-1]
    usage = f'''Usage: {script} CALIB_DIR

    CALIB_DIR must contain merged stacks of flat and dark field images.
    They should be written by "0_merge_frames.py --all", and they must
    be named "Flat.tif" and "Dark.tif". (input)

    Produces "calibration.npz" alongside them, containing the consensus
    flat and dark frames and the per-pixel gain.

    Optional: If a text file named "frames_to_skip.txt" exists and has
              one frame number per line, those frames will be excluded
              from the composite.
    '''
    print(usage, file=sys.stderr)
    sys.exit(1)

import numpy as np

from util import file_exists
from util import load_stack
from util import save_stack

# Input files, from "0_merge_frames.py --all":
FLAT_TIFF = 'Flat.tif'
DARK_TIFF = 'Dark.tif'

# Output file:
NPZ_FILE = 'calibration.npz'

# Output file for debugging:
SKIP_FILE = 'frames_to_skip.txt'
SKIP_TIFF = 'Skip.tif'

calib_dir = os.path.abspath(calib_dir)

flat_tiff = f'{calib_dir}/{FLAT_TIFF}'
dark_tiff = f'{calib_dir}/{DARK_TIFF}'
skip_tiff = f'{calib_dir}/{SKIP_TIFF}'

def load_frames_to_skip(filename):
    '''
    Read a list of frame indices from a text file.

    Each line should contain one integer.
    '''
    with open(filename) as f:
        lines = f.readlines()

    frames = sorted(int(l.strip()) for l in lines)
    return frames

# Load the merged flat and dark stacks.
# Note: the epi version extracts these from an ND2 and writes them out here.
#       On the spinning disk they are the input, so they are not rewritten.
print('Loading flat and dark tiff stacks.')
print(f' - {flat_tiff}')
print(f' - {dark_tiff}')
flat = load_stack(flat_tiff)
dark = load_stack(dark_tiff)

# Remove bad frames. (if any are annotated)
skip_file = f'{calib_dir}/{SKIP_FILE}'

if file_exists(skip_file):
    print('Found a list of frames to skip.')
    print(f' - {skip_file}')

    frames = load_frames_to_skip(skip_file)

    if len(frames) < 1:
        print('Skip file is empty. Retaining all frames.')
    else:
        print(f'Removing the following {len(frames)} frames.')
        print(f' - {frames}')

        # Frames are 1-indexed in Fiji.
        # Switch to 0-indexing for Numpy.
        frames = np.array(frames) - 1

        # Save dropped frames as a TIFF stack.
        print('Saving skipped frame tiff stack.')
        print(f' - {skip_tiff}')
        save_stack(skip_tiff, flat[frames])

        dark = np.delete(dark, frames, axis=0)
        flat = np.delete(flat, frames, axis=0)

# Average to get consensus dark and flat frames.
F = flat.mean(axis=0)
D = dark.mean(axis=0)

# Calculate per-pixel gain.
G = (F - D).mean() / (F - D)

# To correct an image:
# C = (im - D) * G

# Write to disk.
calib_npz = f'{calib_dir}/{NPZ_FILE}'
print('Saving calibration data.')
print(f' - {calib_npz}')
np.savez(calib_npz, Flat=F, Dark=D, Gain=G)
