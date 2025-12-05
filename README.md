# z-ring-constriction

Z-ring constriction dynamics from time-lapse microscopy

## Overview

This pipeline measures the rate of Z-ring constriction and condensation during bacterial cell division in live cells. It is designed for cells expressing fluorescently tagged FtsZ, imaged via spinning disk confocal time-lapse microscopy.

Two main challenges arise:

1. Z-rings are randomly oriented, due to the randomness of cell mounting.
2. Z-rings drift during imaging, as cells double in size during each round of division.

To handle this, the pipeline tracks and registers each individual Z-ring, detects its orientation, extracts a kymograph, and merges all events into a composite for measurement.

**Note:** Fluorescent tags on FtsZ disrupt polymer function, likely affecting constriction dynamics. See Appendices B and E of the linked dissertation for discussion.

## Pipeline

The analysis proceeds through eight stages:

0. **Calibrate** — Generate flat-field correction matrices from dark and fluorescein calibration images.
1. **Merge frames** — Combine multi-position acquisitions into single TIFF stacks.
2. **Flatten and register** — Apply flat-field correction and frame-to-frame registration.
3. **Detect rings** — Identify and track Z-rings across frames using TrackMate (Fiji/Jython).
4. **Parse tracks** — Extract spot and track data from TrackMate XML output.
5. **Register rings** — Crop and register individual Z-ring time series.
6. **Find division plane** — Estimate ring orientation by maximizing FWHM along constriction axes.
7. **Find constriction timing** — Extract kymographs and identify constriction start/end from ring width dynamics.
8. **Make composites** — Average multiple constriction events into composite kymographs.

Supporting modules (`kymo.py`, `fwhm.py`, `util.py`) provide kymograph generation, FWHM calculation, and I/O utilities.

## Dependencies

- Python 3.8+
- pandas
- numpy
- scipy
- matplotlib
- aicsimageio
- pystackreg
- Fiji with TrackMate plugin (for step 3)

## Citation

Developed during doctoral research documented in:

> Mallard, W. (2025). *FtsZ phosphorylation modulates tail-core binding to tune cell division in Bacillus subtilis*. Doctoral dissertation, Harvard University.

## License

MIT License. See [LICENSE](LICENSE).
