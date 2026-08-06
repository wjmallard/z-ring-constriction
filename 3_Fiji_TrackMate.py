#@ File (label="Input directory", style="directory") basedir

# To set basedir:
# - When run in the GUI, Fiji presents a directory chooser.
# - When run headless, Fiji reads it from the command line.
#
#   ImageJ-macosx \
#   --headless \
#   --console \
#   --run 3_Fiji_TrackMate.py 'basedir="/path/to/data"'

import java.io.File as File

from loci.plugins import BF
from loci.plugins.in import ImporterOptions

from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Logger

from fiji.plugin.trackmate.io import TmXmlWriter

from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking.kalman import KalmanTrackerFactory
from fiji.plugin.trackmate.features import FeatureFilter

import os
from glob import glob

INPUT_PATTERN = '*.registered.tif'

# Note: The TrackMate GUI asks for *diameter* in pixels.
#       The TrackMate API asks for *radius* in pixels.
#
# Spot radius:
#  - 100x: 6 pixels
#  - 60x: 4 pixels
#
# Quality threshold:
#  - 100x: 20
#  - 60x: 40
DetectorSettings = {
    'TARGET_CHANNEL': 1,
    'RADIUS': 4.,
    'THRESHOLD': 40.,
    'DO_MEDIAN_FILTERING': False,
    'DO_SUBPIXEL_LOCALIZATION': False,
}

# Note: The TrackMate GUI asks for *diameter* in pixels.
#       The TrackMate API asks for *radius* in pixels.
#
# Linking distance:
#  - 100x: 2 pixels
#  - 60x: 2 pixels
#
# Search radius:
#  - 100x: 2 pixels
#  - 60x: 2 pixels
#
# Frame gap:
#  - 100x: 2
#  - 60x: 1
TrackerSettings = {
    'LINKING_MAX_DISTANCE': 2.,
    'KALMAN_SEARCH_RADIUS': 2.,
    'MAX_FRAME_GAP': 1,
}

SpotQualityThreshold = 40.

SpotFilters = {
}

TrackFilters = {
    'NUMBER_SPOTS': 8,
}

def select_stacks():
    '''
    Find stacks to process in the input directory.
    '''
    pattern = os.path.join(str(basedir), INPUT_PATTERN)
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

def run_TrackMate(filename):
    '''
    Run TrackMate on a single stack.
    '''
    #
    # Open image and configure Model.
    #
    im = open_bioformats(filename)

    model = Model()
    model.setLogger(Logger.VOID_LOGGER)

    logger = model.getLogger()

    settings = Settings(im)
    settings.addAllAnalyzers()

    #
    # Configure detector.
    #
    settings.detectorFactory = LogDetectorFactory()
    settings.detectorSettings = DetectorSettings

    #
    # Configure spot filters.
    #
    # Args: feature, value, isAbove
    settings.initialSpotFilterValue = SpotQualityThreshold
    for k, v in SpotFilters.items():
        filt = FeatureFilter(k, v, True)
        settings.addSpotFilter(filt)

    #
    # Configure tracker.
    #
    settings.trackerFactory = KalmanTrackerFactory()
    settings.trackerSettings = TrackerSettings

    #
    # Configure track filters.
    #
    # Args: feature, value, isAbove
    for k, v in TrackFilters.items():
        filt = FeatureFilter(k, v, True)
        settings.addTrackFilter(filt)

    #
    # Run TrackMate.
    #
    trackmate = TrackMate(model, settings)
    trackmate.checkInput()
    trackmate.process()

    #
    # Save results to XML file.
    #
    im_info = im.getOriginalFileInfo()
    xml_path = im_info.directory
    xml_filename = im_info.fileName.rsplit('.')[0] + '.xml'
    outFile = File(xml_path, xml_filename)

    writer = TmXmlWriter(outFile, logger)
    writer.appendModel(trackmate.getModel())
    writer.appendSettings(trackmate.getSettings())
    writer.writeToFile()

#
# Main()
#
stacks = select_stacks()

for n, src in enumerate(stacks):
    print '[%d/%d] %s' % (n+1, len(stacks), src)
    run_TrackMate(src)

print 'Tracking complete.'
