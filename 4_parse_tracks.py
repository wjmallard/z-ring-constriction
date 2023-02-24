#!/usr/bin/env python
import sys
try:
    xml_files = sys.argv[1:]
    assert xml_files
except:
    script = sys.argv[0].split('/')[-1]
    usage = f'''Usage: {script} TrackMate.xml'''
    print(usage, file=sys.stderr)
    sys.exit(1)

import pandas as pd
import xml.etree.cElementTree as et

def parse_tracks(xml_file):
    #
    # Extract spots from each frame of the tiff stack.
    #
    Frames = [
        elem
        for _, elem in et.iterparse(xml_file)
        if elem.tag == 'AllSpots'
    ]
    Frames = Frames[0]

    spot_fields = [
        'ID',
        'FRAME',
        'POSITION_X',
        'POSITION_Y',

        'POSITION_Z',
        'POSITION_T',
        'RADIUS',
        'QUALITY',
        'VISIBILITY',

        'TOTAL_INTENSITY_CH1',
        'MIN_INTENSITY_CH1',
        'MAX_INTENSITY_CH1',
        'MEDIAN_INTENSITY_CH1',
        'MEAN_INTENSITY_CH1',
        'STD_INTENSITY_CH1',
        'CONTRAST_CH1',
        'SNR_CH1',
    ]

    Spots = [
        [
            Spot.get(field)
            for field in spot_fields
        ]
        for Frame in Frames
        for Spot in Frame.iterfind('Spot')
    ]

    Spots = pd.DataFrame(
        data=Spots,
        columns=spot_fields,
    )
    Spots = Spots.rename(columns={
        'ID': 'SPOT_ID',
    })
    Spots['SPOT_ID'] = Spots.SPOT_ID.astype(int)
    Spots['FRAME'] = Spots.FRAME.astype(int)

    #
    # Extract assignments of spots to tracks.
    #
    Tracks = [
        elem
        for _, elem in et.iterparse(xml_file)
        if elem.tag == 'AllTracks'
    ]
    Tracks = Tracks[0]

    Edges = [
        (
            Track.get('name'),
            Track.get('TRACK_ID'),
            Edge.get(field),
        )
        for Track in Tracks
        for Edge in Track
        for field in [
            'SPOT_SOURCE_ID',
            'SPOT_TARGET_ID'
        ]
    ]

    Edges = pd.DataFrame(
        data=Edges,
        columns=[
            'Track_Name',
            'TRACK_ID',
            'SPOT_ID'
        ]
    )
    Edges['TRACK_ID'] = Edges.TRACK_ID.astype(int)
    Edges['SPOT_ID'] = Edges.SPOT_ID.astype(int)

    #
    # Merge spot and track tables.
    #
    Track_Spots = pd.merge(Edges, Spots, on='SPOT_ID')
    Track_Spots = Track_Spots.drop_duplicates()
    Track_Spots = Track_Spots.sort_values(by=['TRACK_ID', 'FRAME'])
    Track_Spots = Track_Spots.reset_index(drop=True)

    #
    # Extract track metadata.
    #
    Track_Metadata = [Track.attrib
                      for Track in Tracks.iterfind('Track')]
    Track_Metadata = pd.DataFrame(Track_Metadata)
    Track_Metadata = Track_Metadata.rename(columns={'name': 'Track_Name'})

    #
    # Write to disk.
    #
    assert xml_file.endswith('.xml')
    basename = xml_file[:-len('.xml')]

    Track_Spots.to_csv(basename + '.TrackSpots.tsv', sep='\t', index=None)
    Track_Metadata.to_csv(basename + '.TrackMetadata.tsv', sep='\t', index=None)


for n, filename in enumerate(xml_files):

    print(f'[{n+1}/{len(xml_files)}] {filename}')

    parse_tracks(filename)
