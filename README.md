## ESPA Reprojection Version 1.0.2 - Release Notes

See git tag [v1.0.2]

This project contains application source code for warping/reprojecting ESPA Raw Binary (ESPA's internal) formatted data.  This application is intended to be used by ESPA Processing.  This application does not perform the reprojection, it is a wrapper around GDAL's gdalwarp.

## Release Notes
* Version change
* Only perform validation on image bands

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


#### Support Information

This project is unsupported software provided by the U.S. Geological Survey (USGS) Earth Resources Observation and Science (EROS) Land Satellite Data Systems (LSDS) Project. For questions regarding products produced by this source code, please contact us at custserv@usgs.gov.

#### Disclaimer

This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software has not received final approval by the U.S. Geological Survey (USGS). No warranty, expressed or implied, is made by the USGS or the U.S. Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.


[2]: https://landsat.usgs.gov/contact
    
