from ij import IJ, WindowManager
from ij.io import DirectoryChooser
from ij.plugin import Zoom
from ij.plugin.frame import RoiManager

from loci.plugins import BF
from loci.plugins.in import ImporterOptions

import os
from glob import glob
import time

INPUT_PATTERN = '*_s1__registered.Track_*.tif'
IMAGE_ZOOM = 8  # x 100%
LOOP_DELAY = .1  # seconds

def select_stacks():
    '''
    Prompt user for a directory to process.
    '''
    basedir = DirectoryChooser(None).getDirectory()
    if basedir is None:
        print 'No directory selected. Aborting.'
        return []

    pattern = os.path.join(basedir, INPUT_PATTERN)
    targets = sorted(glob(pattern))

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

def annotate_image(filename):
    '''
    Manually annotate a Z-ring by drawing an ROI.
    '''
    # Open the image.
    im = open_bioformats(filename)

    # Construct output file paths.
    im_info = im.getOriginalFileInfo()
    out_path = im_info.directory
    basename = im_info.fileName.rsplit('.', 1)[0]

    outFile = '%s/%s.end' % (out_path, basename)
    roiFile = '%s/%s.roi' % (out_path, basename)

    # Skip images with existing ROIs.
    if os.path.exists(outFile):
        print ' - Image already annotated. Skipping.'
        return
    if not os.path.exists(roiFile):
        print ' - Image has no ROI file. Skipping.'
        return

    # Initialize the ROI Manager.
    rm = RoiManager.getInstance()
    if (rm is None):
        rm = RoiManager()
    rm.reset()

    # Display the image.
    im.show()
    Zoom.set(im, IMAGE_ZOOM)
    IJ.setTool('line')
    
    # Load existing ROI.
    rm.open(roiFile)
    rm.select(0)

    # Event loop:
    # Wait for an ROI selection.
    # Save the current frame as the end of constriction.
    while True:

        time.sleep(LOOP_DELAY)

        if rm.getCount() > 1:

            t_end = im.getFrame()
            with open(outFile, 'w') as fid:
                print >> fid, str(t_end)
            
            print ' - Saved ring constriction end time. (%d)' % t_end
            rm.reset()
            break

    # Close the image.
    im.close()

#
# Main()
#
stacks = select_stacks()

for n, src in enumerate(stacks):
    print '[%d/%d] %s' % (n+1, len(stacks), src)
    annotate_image(src)

print 'Annotation complete.'
