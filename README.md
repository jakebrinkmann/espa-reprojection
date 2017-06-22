## ESPA Reprojection Version 1.0.0 - Release Notes

See git tag [v1.0.0]

This project contains application source code for warping/reprojecting ESPA Raw Binary (ESPA's internal) formatted data.  This application is intended to be used by ESPA Processing.  This application does not perform the reprojection, it is a wrapper around GDAL's gdalwarp.

## Support Information
This project is unsupported software provided by the US Geological Survey (USGS) Earth Resources Observation and Science (EROS) Land Satellite Data Systems (LSDS) Science Research and Development (LSRD) Project.  For questions regarding products produced by this source code, please contact the Landsat Contact Us page and specify USGS CDR/ECV in the "Regarding" section. https://landsat.usgs.gov/contactus.php

## Release Notes
* Initial implementation derived from original implementation within ESPA Processing

## Installation

### Dependencies
* Python 2.7.X and Numpy/GDAL
* [espa-python-library](https://github.com/USGS-EROS/espa-python-library) >= 1.1.0
* [GDAL](http://www.gdal.org/) 1.11.1
  - The command line tools are utilized for some of the processing steps.

### Build Steps
```
make install
```
## Usage
See `espa_reprojection.py --help` for command line details.

### Data Processing Requirements
This version of the Reprojection application requires the input for reprojection to be in the ESPA Raw Binary (ESPA's internal) format.

### Data Postprocessing
After compiling the [espa-product-formatter](https://github.com/USGS-EROS/espa-product-formatter) libraries and tools, the `convert_espa_to_gtif`, `convert_espa_to_hdf` and `convert_espa_to_netcdf` command-line tools can be used to convert the ESPA internal file format to HDF, GeoTIFF, or NetCDF.  Otherwise the data will remain in the ESPA internal file format, which includes each band in the ENVI file format (i.e. raw binary file with associated ENVI header file) and an overall XML metadata file.
