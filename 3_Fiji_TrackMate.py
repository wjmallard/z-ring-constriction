from ij import IJ
from ij.io import DirectoryChooser

import java.io.File as File

from loci.plugins import BF
from loci.plugins.in import ImporterOptions

from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Logger

from fiji.plugin.trackmate.util import TMUtils
from fiji.plugin.trackmate.io import TmXmlWriter

from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking.kalman import KalmanTrackerFactory
from fiji.plugin.trackmate.features import FeatureFilter

import os
from glob import glob

INPUT_PATTERN = '*__registered.tif'

DetectorSettings = {
    'TARGET_CHANNEL': 1,
    'RADIUS': 6.,
    'THRESHOLD': 20.,
    'DO_MEDIAN_FILTERING': False,
    'DO_SUBPIXEL_LOCALIZATION': False,
}

TrackerSettings = {
    'LINKING_MAX_DISTANCE': 2.,
    'KALMAN_SEARCH_RADIUS': 2.,
    'MAX_FRAME_GAP': 2,
}

SpotQualityThreshold = 20.

SpotFilters = {
}

TrackFilters = {
    'NUMBER_SPOTS': 10,
}

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

def log_startup(logger, settings):
    '''
    Adapted from:
    TrackMate/src/main/java/fiji/plugin/trackmate/gui/wizard/descriptors/StartDialogDescriptor.java
    '''
    # aboutToDisplayPanel():
    PUB1_URL = "https://doi.org/10.1101/2021.09.03.458852"

    PUB1_TXT = ("Ershov D, Phan M-S, Pylvanainen JW, Rigaud SU, et al. "
                "'Bringing TrackMate in the era of machine-learning and deep-learning.'"
                "bioRxiv. 2021; doi:10.1101/2021.09.03.458852")

    welcomeMessage = (TrackMate.PLUGIN_NAME_STR + " v" + TrackMate.PLUGIN_NAME_VERSION
                      + " started on:\n" + TMUtils.getCurrentTimeString())

    logger.log(welcomeMessage)
    logger.log("Please note that TrackMate is available through Fiji, and is based on a publication. "
               "If you use it successfully for your research please be so kind to cite our work:")
    logger.log(PUB1_TXT)
    logger.log(PUB1_URL)
    logger.log("and / or:")
    logger.log("Tinevez, JY.; Perry, N. & Schindelin, J. et al. (2017), "
               "'TrackMate: An open and extensible platform for single-particle tracking.', "
               "Methods 115: 80-90, PMID 27713081.")
    logger.log("https://www.sciencedirect.com/science/article/pii/S1046202316303346")

    logger.log("\nNumerical feature analyzers:")
    logger.log(settings.toStringFeatureAnalyzersInfo())

    # aboutToHidePanel():
    logger.log("Image region of interest:")
    logger.log(settings.toStringImageInfo())

def log_detector(logger, settings):
    '''
    Adapted from:
    TrackMate/src/main/java/fiji/plugin/trackmate/gui/wizard/descriptors/SpotDetectorDescriptor.java
    '''
    # aboutToHidePanel():
    logger.log("Configured detector "
               + settings.detectorFactory.getName()
               + " with settings:")
    logger.log(TMUtils.echoMap(settings.detectorSettings, 2))

def log_tracker(logger, settings):
    '''
    Adapted from:
    TrackMate/src/main/java/fiji/plugin/trackmate/gui/wizard/descriptors/SpotTrackerDescriptor.java
    '''
    # aboutToHidePanel():
    logger.log("Configured tracker "
               + settings.trackerFactory.getName()
               + " with settings:")
    logger.log(TMUtils.echoMap(settings.trackerSettings, 2))

def run_TrackMate(filename):
    '''
    Run TrackMate on a single stack.
    '''
    #
    # Open image and configure Model.
    #
    im = open_bioformats(filename)

    model = Model()
    model.setLogger(Logger.IJ_LOGGER)

    logger = model.getLogger()
    logger.log('\Clear')

    settings = Settings(im)
    settings.addAllAnalyzers()

    log_startup(logger, settings)

    #
    # Configure detector.
    #
    settings.detectorFactory = LogDetectorFactory()
    settings.detectorSettings = DetectorSettings
    log_detector(logger, settings)

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
    log_tracker(logger, settings)

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
    writer.appendLog(IJ.getLog())
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
