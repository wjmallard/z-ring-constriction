from ij import IJ, WindowManager
from ij.io import DirectoryChooser
from ij.plugin import Zoom
from ij.plugin.frame import RoiManager

import java.io.File as File

from loci.plugins import BF
from loci.plugins.in import ImporterOptions

import os
from glob import glob
import time

INPUT_PATTERN = 'test/*__registered.Track_*.tif'
IMAGE_ZOOM = 8

def select_stacks():
    '''
    Prompt user for a directory to process.
    '''
    basedir = DirectoryChooser(None).getDirectory()
    if basedir is None:
        print 'No directory selected. Aborting.'
        return []

    pattern = os.path.join(basedir, INPUT_PATTERN)
    targets = glob(pattern)

    print 'Found %s images matching:' % len(targets)
    print ' - %s' % pattern

    return targets

def open_bioformats(filename, channel=0, frame=-1, series=0):
    '''
    Open an ND2 file via BioFormats.

    Channels and Series are 0-indexed in Jython,
    even though their names are 1-indexed in Fiji.

    c=0 --> "C1"
    s=0 --> "Series 01"

    Series are off by default; use setSeriesOn().
    '''
    options = ImporterOptions()

    options.setId(filename)
    options.setColorMode('Grayscale')
    options.setCBegin(series, channel)
    options.setCEnd(series, channel)
    options.setSeriesOn(series, True)

    if frame >= 0:
        options.setTBegin(series, frame)
        options.setTEnd(series, frame)

    imps = BF.openImagePlus(options)

    return imps[0]

def process_image(filename):

    print 'Processing:', filename

    # Open the image.
    im = open_bioformats(filename)

    im_info = im.getOriginalFileInfo()
    out_path = im_info.directory
    out_name = im_info.fileName.rsplit('.', 1)[0] + '.roi'
    outFile = File(out_path, out_name)

    # Skip images with existing ROIs.
    if outFile.exists():
        print '%s: ROI already exists. Skipping.' % outFile
        return

    # Display the image.
    im.show()
    Zoom.set(im, IMAGE_ZOOM)

    # Initialize the ROI Manager.
    rm = RoiManager.getInstance()
    if (rm is None):
        rm = RoiManager()
    rm.reset()

    # Event loop:
    # Wait for an ROI selection, and save it.
    # Or, if the user closes the image, skip it.
    while True:

        time.sleep(.1)

        if rm.getCount() > 0:
            print 'Saving ROI.'
            rm.select(0)
            rm.save(outFile.path)
            rm.delete(0)
            break

        if WindowManager.getImageCount() < 1:
            print 'Image closed. Skipping ROI.'
            return

    # Close the image.
    im.close()

#
# Main()
#
stacks = select_stacks()

for n, src in enumerate(stacks):
    process_image(src)
