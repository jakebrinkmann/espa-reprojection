#! /usr/bin/env python

'''
Description: Alters product extents, projections and pixel sizes.

License: NASA Open Source Agreement 1.3
'''

import os
import sys
import glob
import logging
import copy
import subprocess
from argparse import ArgumentParser

from lxml import objectify as objectify
from osgeo import gdal, osr
import numpy as np

from espa import Metadata
from cStringIO import StringIO


VERSION = 'espa_reprojection 1.0.1'


# We are only supporting one radius when warping to sinusoidal
SINUSOIDAL_SPHERE_RADIUS = 6371007.181

# We do not allow any user selectable choices for this projection
GEOGRAPHIC_PROJ4_STRING = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

# Some defines for common pixels sizes in decimal degrees
DEG_FOR_30_METERS = 0.0002695
DEG_FOR_15_METERS = (DEG_FOR_30_METERS / 2.0)
DEG_FOR_1_METER = (DEG_FOR_30_METERS / 30.0)

# Supported datums - the strings for them
WGS84 = 'WGS84'
NAD27 = 'NAD27'
NAD83 = 'NAD83'
# WGS84 should always be first in the list
VALID_DATUMS = [WGS84, NAD27, NAD83]


logger = None


class LoggingFilter(logging.Filter):
    """Forces 'ESPA' to be provided in the 'subsystem' tag of the log format
       string
    """

    def filter(self, record):
        """Provide the string for the 'subsystem' tag"""

        record.subsystem = 'ESPA'

        return True


class ExceptionFormatter(logging.Formatter):
    """Modifies how exceptions are formatted
    """

    def formatException(self, exc_info):
        """Specifies how to format the exception text"""

        result = super(ExceptionFormatter, self).formatException(exc_info)

        return repr(result)

    def format(self, record):
        """Specifies how to format the message text if it is an exception"""

        s = super(ExceptionFormatter, self).format(record)
        if record.exc_text:
            s = s.replace('\n', ' ')
            s = s.replace('\\n', ' ')

        return s


def setup_logging(args):
    """Configures the logging/reporting

    Args:
        args <args>: Command line arguments
    """

    global logger

    # Setup the logging level
    logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG

    handler = logging.StreamHandler(sys.stdout)
    formatter = ExceptionFormatter(fmt=('%(asctime)s.%(msecs)03d'
                                        ' %(subsystem)s'
                                        ' %(levelname)-8s'
                                        ' [%(filename)s:'
                                        '%(lineno)d]'
                                        ' %(message)s'),
                                   datefmt='%Y-%m-%dT%H:%M:%S')

    handler.setFormatter(formatter)
    handler.addFilter(LoggingFilter())

    logger = logging.getLogger()
    logger.setLevel(logging_level)
    logger.addHandler(handler)


def bcl_add_customization_arguments(sub_p, required, pixel_units):
    """Adds the customization arguments to the parser

    Args:
        sub_p <ArgumentParser>: Parent parser
        required <ArgumentParser>: Required group for any required arguments
        pixel_units <list>: Pixel unit choices for the projection
    """

    # Future options ???? ['cubicspline', 'lanczos']
    required.add_argument('--resample-method',
                        action='store',
                        dest='resample_method',
                        required=True,
                        choices=['near', 'bilinear', 'cubic'],
                        default='near',
                        metavar='TEXT',
                        type=str,
                        help='Resampling method to use')

    custom = sub_p.add_argument_group('pixel size arguments')
    custom.add_argument('--pixel-size',
                        action='store',
                        dest='pixel_size',
                        default=None,
                        metavar='FLOAT',
                        type=float,
                        help='Pixel size for the output product')

    custom.add_argument('--pixel-size-units',
                        action='store',
                        dest='pixel_size_units',
                        choices=pixel_units,
                        default=None,
                        metavar='TEXT',
                        type=str,
                        help='Units for the pixel size')

    custom = sub_p.add_argument_group('extent arguments')
    custom.add_argument('--extent-minx',
                        action='store',
                        dest='extent_minx',
                        default=None,
                        metavar='FLOAT',
                        type=float,
                        help='Minimum X direction extent value')

    custom.add_argument('--extent-maxx',
                        action='store',
                        dest='extent_maxx',
                        default=None,
                        metavar='FLOAT',
                        type=float,
                        help='Maximum X direction extent value')

    custom.add_argument('--extent-miny',
                        action='store',
                        dest='extent_miny',
                        default=None,
                        metavar='FLOAT',
                        type=float,
                        help='Minimum Y direction extent value')

    custom.add_argument('--extent-maxy',
                        action='store',
                        dest='extent_maxy',
                        default=None,
                        metavar='FLOAT',
                        type=float,
                        help='Maximum Y direction extent value')

    custom.add_argument('--extent-units',
                        action='store',
                        dest='extent_units',
                        choices=['meters', 'dd'],
                        default=None,
                        metavar='TEXT',
                        type=str,
                        help='Units for the extent')

    custom = sub_p.add_argument_group('output format arguments')
    custom.add_argument('--output-format',
                        action='store',
                        dest='output_format',
                        default='envi',
                        metavar='TEXT',
                        type=str,
                        help='output format supported by GDAL')


def bcl_add_central_meridian(parser):
    """Adds the central meridian argument to the parser

    Args:
        parser <ArgumentParser>: Parser to add the argument to
    """

    parser.add_argument('--central-meridian',
                        action='store',
                        dest='central_meridian',
                        required=True,
                        default=None,
                        metavar='FLOAT',
                        type=float,
                        help='Central Meridian reprojection value')


def bcl_add_origin_latitude(parser):
    """Adds the origin latitude argument to the parser

    Args:
        parser <ArgumentParser>: Parser to add the argument to
    """

    parser.add_argument('--origin-latitude',
                        action='store',
                        dest='origin_latitude',
                        required=True,
                        default=None,
                        metavar='FLOAT',
                        type=float,
                        help='Origin Latitude reprojection value')


def bcl_add_datum(parser):
    """Adds the datum argument to the parser

    Args:
        parser <ArgumentParser>: Parser to add the argument to
    """

    parser.add_argument('--datum',
                        action='store',
                        dest='datum',
                        choices=VALID_DATUMS,
                        required=True,
                        default=WGS84,
                        metavar='TEXT',
                        type=str,
                        help='Datum to use')


def bcl_add_false_easting_northing(parser):
    """Adds the false easting and false northing arguments to the parser

    Args:
        parser <ArgumentParser>: Parser to add the argument to
    """

    parser.add_argument('--false-easting',
                        action='store',
                        dest='false_easting',
                        required=True,
                        default=None,
                        metavar='FLOAT',
                        type=float,
                        help='False Easting reprojection value')

    parser.add_argument('--false-northing',
                        action='store',
                        dest='false_northing',
                        required=True,
                        default=None,
                        metavar='FLOAT',
                        type=float,
                        help='False Northing reprojection value')


def bcl_add_none(parser):
    """Adds the None command and parameters to the parser

    Args:
        parser <ArgumentParser>: Parser to add the command to
    """

    description = 'Only Customization'
    sub_p = parser.add_parser('none',
                              description=description,
                              help=description)

    required = sub_p.add_argument_group('required arguments')

    bcl_add_customization_arguments(sub_p, required, ['meters'])


def bcl_add_proj4(parser):
    """Adds the Proj4 command and parameters to the parser

    Args:
        parser <ArgumentParser>: Parser to add the command to
    """

    description = 'PROJ4 Projection String'
    sub_p = parser.add_parser('proj4',
                              description=description,
                              help=description)

    required = sub_p.add_argument_group('required arguments')

    required.add_argument('--proj4-string',
                       action='store',
                       dest='proj4_string',
                       required=True,
                       default=None,
                       metavar='<proj4 string>',
                       help='Specify the projection using a proj4 string')

    bcl_add_customization_arguments(sub_p, required, ['meters', 'dd'])


def bcl_add_lonlat(parser):
    """Adds the Geographic command and parameters to the parser

    Args:
        parser <ArgumentParser>: Parser to add the command to
    """

    description = 'Geographic Projection'
    sub_p = parser.add_parser('lonlat',
                              description=description,
                              help=description)

    required = sub_p.add_argument_group('required arguments')

    bcl_add_customization_arguments(sub_p, required, ['dd'])


def bcl_add_utm(parser):
    """Adds the UTM command and parameters to the parser

    Args:
        parser <ArgumentParser>: Parser to add the command to
    """

    description = 'UTM Projection'
    sub_p = parser.add_parser('utm',
                              description=description,
                              help=description)

    required = sub_p.add_argument_group('required arguments')

    required.add_argument('--north-south',
                       action='store',
                       dest='north_south',
                       required=True,
                       choices=['north', 'south'],
                       default=None,
                       type=str,
                       help='UTM North or South')

    required.add_argument('--zone',
                       action='store',
                       dest='zone',
                       required=True,
                       default=None,
                       metavar='INT',
                       type=int,
                       help='UTM Zone value')

    bcl_add_customization_arguments(sub_p, required, ['meters'])


def bcl_add_sinu(parser):
    """Adds the Sinusoidal command and parameters to the parser

    Args:
        parser <ArgumentParser>: Parser to add the command to
    """

    description = 'Sinusoidal Projection'
    sub_p = parser.add_parser('sinu',
                              description=description,
                              help=description)

    required = sub_p.add_argument_group('required arguments')

    bcl_add_central_meridian(required)
    bcl_add_false_easting_northing(required)
    bcl_add_customization_arguments(sub_p, required, ['meters'])


def bcl_add_aea(parser):
    """Adds the Albers Equal Area command and parameters to the parser

    Args:
        parser <ArgumentParser>: Parser to add the command to
    """

    description = 'Albers Equal Area Projection'
    sub_p = parser.add_parser('aea',
                              description=description,
                              help=description)

    required = sub_p.add_argument_group('required arguments')

    bcl_add_central_meridian(required)

    required.add_argument('--std-parallel-1',
                       action='store',
                       dest='std_parallel_1',
                       required=True,
                       default=None,
                       metavar='FLOAT',
                       type=float,
                       help='Standard Parallel 1 reprojection value')

    required.add_argument('--std-parallel-2',
                       action='store',
                       dest='std_parallel_2',
                       required=True,
                       default=None,
                       metavar='FLOAT',
                       type=float,
                       help='Standard Parallel 2 reprojection value')

    bcl_add_origin_latitude(required)
    bcl_add_false_easting_northing(required)
    bcl_add_datum(required)
    bcl_add_customization_arguments(sub_p, required, ['meters'])


def bcl_add_ps(parser):
    """Adds the Polar Stereographic command and parameters to the parser

    Args:
        parser <ArgumentParser>: Parser to add the command to
    """

    description = 'Polar-Stereographic Projection'
    sub_p = parser.add_parser('ps',
                              description=description,
                              help=description)

    required = sub_p.add_argument_group('required arguments')

    required.add_argument('--latitude-true-scale',
                       action='store',
                       dest='latitude_true_scale',
                       required=True,
                       default=None,
                       metavar='FLOAT',
                       type=float,
                       help='Latitude True Scale reprojection value')

    required.add_argument('--longitude-pole',
                       action='store',
                       dest='longitude_pole',
                       required=True,
                       default=None,
                       metavar='FLOAT',
                       type=float,
                       help='Longitude Pole reprojection value')

    bcl_add_origin_latitude(required)
    bcl_add_false_easting_northing(required)
    bcl_add_customization_arguments(sub_p, required, ['meters'])


def build_command_line_arguments():
    """Read arguments from the command line

    Returns:
        <args>: The arguments read from the command line
    """

    description = ('Reproject the data defined in the ESPA Raw Binary'
                   ' Format to the specified projection')

    parser = ArgumentParser(description=description)

    parser.add_argument('--xml',
                        action='store',
                        dest='xml_filename',
                        required=True,
                        metavar='FILE',
                        help='The XML metadata file to use')

    parser.add_argument('--version',
                        action='version',
                        version=VERSION)

    parser.add_argument('--debug',
                        action='store_true',
                        dest='debug',
                        default=False,
                        help='display error information')

    subparsers = parser.add_subparsers(dest='projection')

    bcl_add_none(subparsers)
    bcl_add_proj4(subparsers)
    bcl_add_lonlat(subparsers)
    bcl_add_utm(subparsers)
    bcl_add_sinu(subparsers)
    bcl_add_aea(subparsers)
    bcl_add_ps(subparsers)

    return parser


class CommandLineError(Exception):
    pass


def verify_command_line(args):
    """Additional verification of command line parameters

    Args:
        args <ArgumentParser>: User supplied command line arguments
    """

    if (args.extent_minx or args.extent_miny or args.extent_maxx
            or args.extent_maxy or args.extent_units):
        if (not args.extent_minx or not args.extent_miny
                or not args.extent_maxx or not args.extent_maxy
                or not args.extent_units):
            raise CommandLineError('All extent arguments must be specified')

    if args.pixel_size or args.pixel_size_units:
        if not args.pixel_size or not args.pixel_size_units:
            raise CommandLineError('All pixel size arguments must be'
                                   ' specified')


def determine_resample_method(args, band):
    """Determines the correct resampling method based on the category of the
       data to be processed

    Args:
        args <ArgumentParser>: User supplied command line arguments
        band <XML object>: Band information from the XML metadata

    Returns:
        <str>: Resample method to use
    """

    # Always use near for qa bands
    if band.attrib['category'] == 'qa':
        return 'near'
    else:
        return args.resample_method


def determine_pixel_size(args, global_metadata, band):
    """Determines the correct pixel size to use based on the band being
       processed

    Args:
        args <ArgumentParser>: User supplied command line arguments
        global_metadata <XML object>: Global metadata from the XML metadata
        band <XML object>: Band information from the XML metadata

    Returns:
        <float>: Pixel size value
    """

    pixel_size = args.pixel_size

    # EXECUTIVE DECISION(Calli)
    # - If the band is (Landsat 7 or 8) and Band 8 do not resize the pixels
    if ((global_metadata.satellite == 'LANDSAT_7' or
            global_metadata.satellite == 'LANDSAT_8') and
            band.attrib['name'] == 'b8'):

        if args.projection == 'lonlat':
            pixel_size = DEG_FOR_15_METERS
        else:
            pixel_size = float(band.pixel_size.attrib['x'])

    return pixel_size


def determine_no_data_value(img_filename):
    """Determines the correct no data value to use based on the band being
       processerd

    Returns:
        <str>: String version of the no data value from the band data
    """

    # Open the image to read the no data value out since the internal
    # ENVI driver for GDAL does not output it, even if it is known
    ds = gdal.Open(img_filename)
    if ds is None:
        raise RuntimeError('GDAL failed to open ({})'.format(img_filename))

    ds_band = None
    ds_band = ds.GetRasterBand(1)

    # Save the no data value since gdalwarp does not write it out when
    # using the ENVI format
    no_data_value = ds_band.GetNoDataValue()
    if no_data_value is not None:
        no_data_value = str(no_data_value)

    # Force a freeing of the memory
    del ds_band
    del ds

    return no_data_value


def build_sinu_proj4_string(args):
    """Builds a proj.4 string for sinusoidal

    Args:
        args <ArgumentParser>: User supplied command line arguments

    Example:
      +proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181
      +ellps=WGS84 +datum=WGS84 +units=m +no_defs
    """

    projection = ('+proj=sinu +lon_0={0} +x_0={1} +y_0={2} +a={3} +b={3}'
                  ' +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
                  .format(args.central_meridian, args.false_easting,
                          args.false_northing, SINUSOIDAL_SPHERE_RADIUS))

    return projection


def build_albers_proj4_string(args):
    """Builds a proj.4 string for albers equal area

    Args:
        args <ArgumentParser>: User supplied command line arguments

    Example:
      +proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0
      +ellps=GRS80 +datum=NAD83 +units=m +no_defs
    """

    projection = ('+proj=aea +lat_1={0} +lat_2={1} +lat_0={2} +lon_0={3}'
                  ' +x_0={4} +y_0={5} +ellps=GRS80 +datum={6} +units=m'
                  ' +no_defs'
                  .format(args.std_parallel_1, args.std_parallel_2,
                          args.origin_latitude, args.central_meridian,
                          args.false_easting, args.false_northing,
                          args.datum))

    return projection


def build_utm_proj4_string(args):
    """Builds a proj.4 string for utm

    Args:
        args <ArgumentParser>: User supplied command line arguments

    Note:
      The ellipsoid probably doesn't need to be specified.

    Examples:
      #### gdalsrsinfo EPSG:32660
      +proj=utm +zone=60 +ellps=WGS84 +datum=WGS84 +units=m +no_defs

      #### gdalsrsinfo EPSG:32739
      +proj=utm +zone=39 +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs
    """

    projection = ''
    if args.north_south == 'north':
        projection = ('+proj=utm +zone={} +ellps=WGS84 +datum=WGS84'
                      ' +units=m +no_defs'.format(args.zone))
    else:
        projection = ('+proj=utm +zone={} +south +ellps=WGS84 +datum=WGS84'
                      ' +units=m +no_defs'.format(args.zone))

    return projection


def build_ps_proj4_string(args):
    """Builds a proj.4 string for polar stereographic
       gdalsrsinfo 'EPSG:3031'

    Args:
        args <ArgumentParser>: User supplied command line arguments

    Examples:
      +proj=stere +lat_0=90 +lat_ts=71 +lon_0=0 +k=1 +x_0=0 +y_0=0
        +datum=WGS84 +units=m +no_defs

      +proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0
        +datum=WGS84 +units=m +no_defs
    """

    projection = ('+proj=stere +lat_ts={0} +lat_0={1} +lon_0={2} +k_0=1.0'
                  ' +x_0={3} +y_0={4} +datum=WGS84 +units=m +no_defs'
                  .format(args.latitude_true_scale, args.origin_latitude,
                          args.longitude_pole, args.false_easting,
                          args.false_northing))

    return projection


def convert_target_projection_to_proj4(args):
    """The target projection is validated against the implemented projections
       and depending on the projection, the correct proj4 parameters are
       returned

    Args:
        args <ArgumentParser>: User supplied command line arguments
    """

    projection = None

    if args.projection == "sinu":
        projection = build_sinu_proj4_string(args)

    elif args.projection == "aea":
        projection = build_albers_proj4_string(args)

    elif args.projection == "utm":
        projection = build_utm_proj4_string(args)

    elif args.projection == "ps":
        projection = build_ps_proj4_string(args)

    elif args.projection == "lonlat":
        projection = GEOGRAPHIC_PROJ4_STRING

    return str(projection)


class TransformPointError(Exception):
    pass


def projection_minbox(args, target_proj4):
    """Determines the minimum box in map coordinates that contains the
       geographic coordinates

    Args:
        args <ArgumentParser>: User supplied command line arguments

    Returns:
        (min_x, min_y, max_x, max_y) in meters

    Note:
        Minimum and maximum extent values are returned in map coordinates.
    """

    logger.debug('Determining Image Extents For Requested Projection')

    # We are always going to be geographic
    source_proj4 = GEOGRAPHIC_PROJ4_STRING

    logger.debug('Using source projection [{}]'.format(source_proj4))
    logger.debug('Using target projection [{}]'.format(target_proj4))

    # Create and initialize the source SRS
    source_srs = osr.SpatialReference()
    source_srs.ImportFromProj4(source_proj4)

    # Create and initialize the target SRS
    target_srs = osr.SpatialReference()
    target_srs.ImportFromProj4(target_proj4)

    # Create the transformation object
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    # Determine the step in decimal degrees
    step = args.pixel_size
    if args.pixel_size_units == 'meters':
        # Convert it to decimal degrees
        step = DEG_FOR_1_METER * args.pixel_size

    # Determine the lat and lon values to iterate over
    longitudes = np.arange(args.extent_minx, args.extent_maxx, step, np.float)
    latitudes = np.arange(args.extent_miny, args.extent_maxy, step, np.float)

    # Initialization using the two corners
    (ul_x, ul_y, z) = transform.TransformPoint(args.extent_minx,
                                               args.extent_maxy)
    if ul_x < 1.0 or ul_y < 1.0:
        raise TransformPointError('Error transforming point')
    (lr_x, lr_y, z) = transform.TransformPoint(args.extent_maxx,
                                               args.extent_miny)
    if lr_x < 1.0 or lr_y < 1.0:
        raise TransformPointError('Error transforming point')

    min_x = min(ul_x, lr_x)
    max_x = max(ul_x, lr_x)
    min_y = min(ul_y, lr_y)
    max_y = max(ul_y, lr_y)

    logger.debug('Direct translation of the provided geographic coordinates')
    logger.debug(', '.join(['min_x', 'min_y', 'max_x', 'max_y']))
    logger.debug(', '.join([str(min_x), str(min_y), str(max_x), str(max_y)]))

    # Walk across the top and bottom of the geographic coordinates
    for lon in longitudes:
        # Upper side
        (ux, uy, z) = transform.TransformPoint(lon, args.extent_maxy)
        if ux < 1.0 or uy < 1.0:
            raise TransformPointError('Error transforming point')

        # Lower side
        (lx, ly, z) = transform.TransformPoint(lon, args.extent_miny)
        if lx < 1.0 or ly < 1.0:
            raise TransformPointError('Error transforming point')

        min_x = min(ux, lx, min_x)
        max_x = max(ux, lx, max_x)
        min_y = min(uy, ly, min_y)
        max_y = max(uy, ly, max_y)

    # Walk along the left and right of the geographic coordinates
    for lat in latitudes:
        # Left side
        (lx, ly, z) = transform.TransformPoint(args.extent_minx, lat)
        if lx < 1.0 or ly < 1.0:
            raise TransformPointError('Error transforming point')

        # Right side
        (rx, ry, z) = transform.TransformPoint(args.extent_maxx, lat)
        if rx < 1.0 or ry < 1.0:
            raise TransformPointError('Error transforming point')

        min_x = min(rx, lx, min_x)
        max_x = max(rx, lx, max_x)
        min_y = min(ry, ly, min_y)
        max_y = max(ry, ly, max_y)

    del transform
    del source_srs
    del target_srs

    logger.debug('Map coordinates after minbox determination')
    logger.debug(', '.join(['min_x', 'min_y', 'max_x', 'max_y']))
    logger.debug(', '.join([str(min_x), str(min_y), str(max_x), str(max_y)]))

    return (min_x, min_y, max_x, max_y)


class InsufficientExtentError(Exception):
    pass


def build_image_extents_string(args, target_proj4):
    """Build the gdalwarp image extents string from the determined min and max
       values

    Returns:
        str('min_x min_y max_x max_y')
    """

    # Nothing to do if we are not sub-setting the data
    if not args.extent_units:
        return None

    # Get the image extents string
    if (args.extent_units == 'dd' and
            (args.projection == 'none' or args.projection != 'lonlat')):

        (min_x, min_y, max_x, max_y) = projection_minbox(args, target_proj4)
    else:
        (min_x, min_y, max_x, max_y) = (args.extent_minx, args.extent_miny,
                                        args.extent_maxx, args.extent_maxy)

    if (max_x - min_x) < args.pixel_size:
        raise InsufficientExtentError('Insufficient output pixels -'
                                      ' longitude direction')
    if (max_y - min_y) < args.pixel_size:
        raise InsufficientExtentError('Insufficient output pixels -'
                                      ' latitude direction')

    return [str(min_x), str(min_y), str(max_x), str(max_y)]


def warp_image(source_file, output_file,
               base_warp_command=None,
               resample_method='near',
               pixel_size=None,
               no_data_value=None):
    """Executes the warping command on the specified source file
    """

    output = ''
    try:
        cmd = copy.deepcopy(base_warp_command)

        # Resample method to use
        cmd.extend(['-r', resample_method])

        # Resize the pixels
        if pixel_size is not None:
            cmd.extend(['-tr', str(pixel_size), str(pixel_size)])

        # Specify the fill/nodata value
        if no_data_value is not None:
            cmd.extend(['-srcnodata', no_data_value])
            cmd.extend(['-dstnodata', no_data_value])

        # Now add the filenames
        cmd.extend([source_file, output_file])

        logger.debug('Warping {0} with {1}'.format(source_file,
                                                   ' '.join(cmd)))
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            logger.error('Warping failed')
            raise

    finally:
        if len(output) > 0:
            logger.debug(output)


class WarpVerificationError(Exception):
    pass


WARP_ERROR_STRING = ('Failed to compute statistics,'
                     ' no valid pixels found in sampling')


def verify_warping(img_filename):

    cmd = ['gdalinfo', '-stats', img_filename]
    logger.debug('Verifying warp with {}'.format(' '.join(cmd)))
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)

        if WARP_ERROR_STRING in output:

            raise WarpVerificationError(WARP_ERROR_STRING)
    except subprocess.CalledProcessError as error:
        logger.error('Warping failed')
        raise


def fix_envi_header(no_data_value, warped_hdr_filename):
    """Update the tmp ENVI header with our own values for some fields
    """

    sb = StringIO()
    with open(warped_hdr_filename, 'r') as warped_fd:
        while True:
            line = warped_fd.readline()
            if not line:
                break
            if (line.startswith('data ignore value') or
                    line.startswith('description')):
                pass
            else:
                sb.write(line)

            if line.startswith('description'):
                # This may be on multiple lines so read lines until
                # we find the closing brace
                if not line.strip().endswith('}'):
                    while 1:
                        next_line = warped_fd.readline()
                        if (not next_line or
                                next_line.strip().endswith('}')):
                            break
                sb.write('description = {ESPA-generated file}\n')
            elif (line.startswith('data type') and
                  (no_data_value is not None)):
                sb.write('data ignore value = %s\n' % no_data_value)

    # Do the actual replace here
    with open(warped_hdr_filename, 'w') as warped_fd:
        warped_fd.write(sb.getvalue())

def replace_product_files(img_filename, hdr_filename,
                          warped_img_filename, warped_hdr_filename):
    """Replace the product files with the newly warped files
    """

    # Remove the original files
    if os.path.exists(img_filename):
        os.unlink(img_filename)
    if os.path.exists(hdr_filename):
        os.unlink(hdr_filename)

    # Rename the temp files back to the original names
    os.rename(warped_img_filename, img_filename)
    os.rename(warped_hdr_filename, hdr_filename)


def determine_pixel_size_units_from_projection(projection_name):
    """Determine the pixel size unit we support for the projection

    Args:
        projection_name <str>: Projection name

    Returns:
        <str>: Unit supported
    """

    if projection_name is not None:
        if projection_name.lower().startswith('transverse_mercator'):
            return 'meters'
        elif projection_name.lower().startswith('polar'):
            return 'meters'
        elif projection_name.lower().startswith('albers'):
            return 'meters'
        elif projection_name.lower().startswith('sinusoidal'):
            return 'meters'
    else:
        # Must be Geographic Projection
        return 'degrees'


def update_band_attributes(band, ds_transform, number_of_lines,
                           number_of_samples, projection_name):
    """Updates the band attributes

    Args:
        band <XML object>: Band XML object to update
        ds_transform <GDAL transform object>: Transformation information
        number_of_lines <int>: Line count
        number_of_samples <int>: Sample count
        projection_name <str>: Project the band is in

    Returns:
        band parameter is updated
    """

    logger.debug('PROJECTION NAME [{}]'.format(projection_name))

    # Update the band information in the XML file
    band.attrib['nlines'] = str(number_of_lines)
    band.attrib['nsamps'] = str(number_of_samples)
    # Need to abs these because they are coming from the transform,
    # which may be correct for the transform,
    # but not how us humans understand it
    band.pixel_size.attrib['x'] = str(abs(ds_transform[1]))
    band.pixel_size.attrib['y'] = str(abs(ds_transform[5]))

    # For sanity report the resample method applied to the data
    logger.debug('RESAMPLE METHOD [{}]'.format(band.resample_method))

    # We only support one unit type for each projection
    band.pixel_size.attrib['units'] = \
        determine_pixel_size_units_from_projection(projection_name)


def update_bands(args, espa_metadata):
    em = objectify.ElementMaker(annotate=False, namespace=None, nsmap=None)

    for band in espa_metadata.xml_object.bands.band:
        img_filename = str(band.file_name)
        logger.info('Updating XML for {}'.format(img_filename))

        resample_method = determine_resample_method(args, band)

        # Update the XML metadata object for the resampling method used
        band.resample_method = \
            em.item(XML_BAND_TRANSLATION[resample_method])

        ds = gdal.Open(img_filename)
        if ds is None:
            msg = 'GDAL failed to open {}'.format(img_filename)
            raise RuntimeError(msg)

        try:
            ds_band = ds.GetRasterBand(1)
            ds_transform = ds.GetGeoTransform()
            ds_srs = osr.SpatialReference()
            ds_srs.ImportFromWkt(ds.GetProjection())
        except Exception:
            raise

        number_of_lines = int(ds_band.YSize)
        number_of_samples = int(ds_band.XSize)

        update_band_attributes(band, ds_transform,
                               number_of_lines, number_of_samples,
                               ds_srs.GetAttrValue('PROJECTION'))

        del ds_band
        del ds


def update_utm_parameters(gm, ds_srs, old_proj_params):
    logger.info('Updating Projection with UTM Parameters')

    # Create an element maker
    em = objectify.ElementMaker(annotate=False, namespace=None, nsmap=None)

    # Get the parameter values
    zone = int(ds_srs.GetUTMZone())

    # Get a new UTM projection parameter object and populate it
    utm_proj_params = em.utm_proj_params()
    utm_proj_params.zone_code = em.item(zone)

    # Add the object to the projection information
    gm.projection_information.replace(old_proj_params, utm_proj_params)

    # Update the attribute values
    gm.projection_information.attrib['projection'] = 'UTM'
    gm.projection_information.attrib['datum'] = WGS84


def update_polar_parameters(gm, ds_srs, old_proj_params):
    logger.info('Updating Projection with Polar Stereographic Parameters')

    # Create an element maker
    em = objectify.ElementMaker(annotate=False, namespace=None, nsmap=None)

    # Get the parameter values
    latitude_true_scale = ds_srs.GetProjParm('latitude_of_origin')
    longitude_pole = ds_srs.GetProjParm('central_meridian')
    false_easting = ds_srs.GetProjParm('false_easting')
    false_northing = ds_srs.GetProjParm('false_northing')

    # Get a new PS projection parameter object and populate it
    ps_proj_params = em.ps_proj_params()
    ps_proj_params.longitude_pole = em.item(longitude_pole)
    ps_proj_params.latitude_true_scale = em.item(latitude_true_scale)
    ps_proj_params.false_easting = em.item(false_easting)
    ps_proj_params.false_northing = em.item(false_northing)

    # Add the object to the projection information
    gm.projection_information.replace(old_proj_params, ps_proj_params)
    # Update the attribute values
    gm.projection_information.attrib['projection'] = 'PS'
    gm.projection_information.attrib['datum'] = WGS84


def update_albers_parameters(args, gm, ds_srs, old_proj_params):
    logger.info('Updating Projection with Albers Equal Area Parameters')

    # Create an element maker
    em = objectify.ElementMaker(annotate=False, namespace=None, nsmap=None)

    # Get the parameter values
    standard_parallel1 = ds_srs.GetProjParm('standard_parallel_1')
    standard_parallel2 = ds_srs.GetProjParm('standard_parallel_2')
    origin_latitude = ds_srs.GetProjParm('latitude_of_center')
    central_meridian = ds_srs.GetProjParm('longitude_of_center')
    false_easting = ds_srs.GetProjParm('false_easting')
    false_northing = ds_srs.GetProjParm('false_northing')

    # Get a new ALBERS projection parameter object and populate it
    albers_proj_params = em.albers_proj_params()
    albers_proj_params.standard_parallel1 = em.item(standard_parallel1)
    albers_proj_params.standard_parallel2 = em.item(standard_parallel2)
    albers_proj_params.central_meridian = em.item(central_meridian)
    albers_proj_params.origin_latitude = em.item(origin_latitude)
    albers_proj_params.false_easting = em.item(false_easting)
    albers_proj_params.false_northing = em.item(false_northing)

    # Add the object to the projection information
    gm.projection_information.replace(old_proj_params, albers_proj_params)

    # Update the attribute values
    gm.projection_information.attrib['projection'] = 'ALBERS'

    # This projection can have different datums, so use the datum
    # requested by the user
    if 'datum' in args:
        gm.projection_information.attrib['datum'] = args.datum
    else:
        gm.projection_information.attrib['datum'] = WGS84


def update_sinu_parameters(gm, ds_srs, old_proj_params):
    logger.info('Updating Projection with Sinusoidal Parameters')

    # Create an element maker
    em = objectify.ElementMaker(annotate=False, namespace=None, nsmap=None)

    # Get the parameter values
    central_meridian = ds_srs.GetProjParm('longitude_of_center')
    false_easting = ds_srs.GetProjParm('false_easting')
    false_northing = ds_srs.GetProjParm('false_northing')

    # Get a new SIN projection parameter object and populate it
    sin_proj_params = em.sin_proj_params()
    sin_proj_params.sphere_radius = em.item(SINUSOIDAL_SPHERE_RADIUS)
    sin_proj_params.central_meridian = em.item(central_meridian)
    sin_proj_params.false_easting = em.item(false_easting)
    sin_proj_params.false_northing = em.item(false_northing)

    # Add the object to the projection information
    gm.projection_information.replace(old_proj_params, sin_proj_params)

    # Update the attribute values
    gm.projection_information.attrib['projection'] = 'SIN'


def convert_imageXY_to_mapXY(image_x, image_y, transform):
    '''
    Description:
      Translate image coordinates into map coordinates
    '''

    map_x = transform[0] + image_x * transform[1] + image_y * transform[2]
    map_y = transform[3] + image_x * transform[4] + image_y * transform[5]

    return (map_x, map_y)


XML_BAND_TRANSLATION = {
    'near': 'nearest neighbor',
    'bilinear': 'bilinear',
    'cubic': 'cubic convolution'
}


# All of these bands can be used for the reference for the reprojected data
# No concern mixing Landsat and Modis since we can only be processing one at
# a time
REFERENCE_BANDS = {
    'L1TP': ['b1', 'b2', 'b3', 'b4', 'b5', 'b7', 'bqa'],
    'level2_qa': ['pixel_qa'],
    'toa_refl': ['toa_band1', 'toa_band2', 'toa_band3', 'toa_band4',
                 'toa_band5', 'toa_band7', 'toa_band1'],
    'sr_refl': ['sr_band1', 'sr_band2', 'sr_band3', 'sr_band4',
                'sr_band5', 'sr_band7', 'sr_band1',
                'sur_refl_b01_1', 'sur_refl_b02_1', 'sur_refl_b03_1',
                'sur_refl_b04_1', 'sur_refl_b05_1', 'sur_refl_b06_1',
                'sur_refl_b07_1',
                'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03',
                'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06',
                'sur_refl_b07'],
    'land_surf_temp': ['LST_Day_1km', 'LST_Night_1km'],
    'spectral_indices': ['250m 16 days NDVI',
                         '250m 16 days EVI',
                         '250m 16 days red reflectance',
                         '250m 16 days NIR reflectance',
                         '250m 16 days blue reflectance',
                         '250m 16 days MIR reflectance',
                         '500m 16 days NDVI',
                         '500m 16 days EVI',
                         '500m 16 days red reflectance',
                         '500m 16 days NIR reflectance',
                         '500m 16 days blue reflectance',
                         '500m 16 days MIR reflectance',
                         '1 km 16 days NDVI',
                         '1 km 16 days EVI',
                         '1 km 16 days red reflectance',
                         '1 km 16 days NIR reflectance',
                         '1 km 16 days blue reflectance',
                         '1 km 16 days MIR reflectance',
                         '1 km monthly NDVI',
                         '1 km monthly EVI',
                         '1 km monthly red reflectance',
                         '1 km monthly NIR reflectance',
                         '1 km monthly blue reflectance',
                         '1 km monthly MIR reflectance']
}


class MissingReferenceBand(Exception):
    pass


def determine_ref_image_filename(espa_metadata):

    img_filename = None

    for band in espa_metadata.xml_object.bands.band:
        product = band.attrib['product']
        name = band.attrib['name']
        if product in REFERENCE_BANDS and name in REFERENCE_BANDS[product]:

            img_filename = str(band.file_name)
            break

    if img_filename is None:
        raise MissingReferenceBand('Required reference bands not found')

    return img_filename


def crosses_antimeridian(args, global_metadata):

    # Determine whether the scene uses lonlat projection and crosses the
    # antimeridian
    if (args.projection == 'lonlat' and
            global_metadata.bounding_coordinates.east < 0 and
            global_metadata.bounding_coordinates.west > 0):
        logger.info('Data crosses the anti-meridian')
        return True

    return False


def update_espa_xml(args, espa_metadata):

    try:
        # Create an element maker
        em = objectify.ElementMaker(annotate=False, namespace=None, nsmap=None)

        update_bands(args, espa_metadata)

        reference_image_filename = determine_ref_image_filename(espa_metadata)
        ds = gdal.Open(reference_image_filename)
        if ds is None:
            msg = 'GDAL failed to open {}'.format(img_filename)
            raise RuntimeError(msg)

        try:
            ds_band = ds.GetRasterBand(1)
            ds_transform = ds.GetGeoTransform()
            ds_srs = osr.SpatialReference()
            ds_srs.ImportFromWkt(ds.GetProjection())
        except Exception:
            raise

        number_of_lines = int(ds_band.YSize)
        number_of_samples = int(ds_band.XSize)
        del ds_band

        ######################################################################
        # Fix the projection information for the warped data
        ######################################################################
        gm = espa_metadata.xml_object.global_metadata

        # Determine whether the scene uses lonlat projection and crosses the
        # antimeridian
        antimeridian_crossing = crosses_antimeridian(args, gm)

        # If the image extents were changed, then the scene center time is
        # meaningless so just remove it
        # We don't have any way to calculate a new one
        if args.extent_units and 'scene_center_time' in gm:
            gm.remove(gm.scene_center_time)

        # Find the projection parameter object from the structure so that it
        # can be replaced with the new one
        # Geographic doesn't have one
        old_proj_params = None
        for item in gm.projection_information.getchildren():
            if 'utm_proj_params' in item.tag:
                old_proj_params = item
                break
            elif 'ps_proj_params' in item.tag:
                old_proj_params = item
                break
            elif 'albers_proj_params' in item.tag:
                old_proj_params = item
                break
            elif 'sin_proj_params' in item.tag:
                old_proj_params = item
                break

        # Rebuild the projection parameters
        projection_name = ds_srs.GetAttrValue('PROJECTION')
        if projection_name is not None:
            if projection_name.lower().startswith('transverse_mercator'):
                update_utm_parameters(gm, ds_srs, old_proj_params)

            elif projection_name.lower().startswith('polar'):
                update_polar_parameters(gm, ds_srs, old_proj_params)

            elif projection_name.lower().startswith('albers'):
                update_albers_parameters(args, gm, ds_srs, old_proj_params)

            elif projection_name.lower().startswith('sinusoidal'):
                update_sinu_parameters(gm, ds_srs, old_proj_params)

        else:
            # Must be Geographic Projection
            logger.info('Updating Projection with Geographic Parameters')
            gm.projection_information.attrib['projection'] = 'GEO'
            gm.projection_information.attrib['datum'] = WGS84
            gm.projection_information.attrib['units'] = 'degrees'
            gm.projection_information.remove(old_proj_params)

        # Fix the UL and LR center of pixel map coordinates
        (map_ul_x, map_ul_y) = convert_imageXY_to_mapXY(0.5, 0.5,
                                                        ds_transform)
        (map_lr_x, map_lr_y) = convert_imageXY_to_mapXY(
            number_of_samples - 0.5, number_of_lines - 0.5, ds_transform)

        # Keep the corner longitudes in the -180..180 range.  GDAL can report
        # corners outside the range in antimeridian crossing cases.
        if antimeridian_crossing:
            if map_ul_x > 180:
                map_ul_x -= 360
            if map_lr_x > 180:
                map_lr_x -= 360
            if map_ul_x < -180:
                map_ul_x += 360
            if map_lr_x < -180:
                map_lr_x += 360

        for cp in gm.projection_information.corner_point:
            if cp.attrib['location'] == 'UL':
                cp.attrib['x'] = str(map_ul_x)
                cp.attrib['y'] = str(map_ul_y)

            if cp.attrib['location'] == 'LR':
                cp.attrib['x'] = str(map_lr_x)
                cp.attrib['y'] = str(map_lr_y)

        # Fix the UL and LR center of pixel latitude and longitude coordinates
        srs_lat_lon = ds_srs.CloneGeogCS()
        coord_tf = osr.CoordinateTransformation(ds_srs, srs_lat_lon)
        for corner in gm.corner:
            if corner.attrib['location'] == 'UL':
                (lon, lat, height) = \
                    coord_tf.TransformPoint(map_ul_x, map_ul_y)

                # Keep the corner longitudes in the -180..180 range.
                if antimeridian_crossing:
                    if lon > 180:
                        lon -= 360
                    if lon < -180:
                        lon += 360

                corner.attrib['longitude'] = str(lon)
                corner.attrib['latitude'] = str(lat)

            if corner.attrib['location'] == 'LR':
                (lon, lat, height) = \
                    coord_tf.TransformPoint(map_lr_x, map_lr_y)

                # Keep the corner longitudes in the -180..180 range.
                if antimeridian_crossing:
                    if lon > 180:
                        lon -= 360
                    if lon < -180:
                        lon += 360

                corner.attrib['longitude'] = str(lon)
                corner.attrib['latitude'] = str(lat)

        # Determine the bounding coordinates
        # Initialize using the UL and LR, then walk the edges of the image,
        # because some projections may not have the values in the corners of
        # the image
        # UL
        (map_x, map_y) = convert_imageXY_to_mapXY(0.0, 0.0, ds_transform)
        (ul_lon, ul_lat, height) = coord_tf.TransformPoint(map_x, map_y)
        # LR
        (map_x, map_y) = convert_imageXY_to_mapXY(number_of_samples,
                                                  number_of_lines,
                                                  ds_transform)
        (lr_lon, lr_lat, height) = coord_tf.TransformPoint(map_x, map_y)

        # Set the initial values
        west_lon = min(ul_lon, lr_lon)
        east_lon = max(ul_lon, lr_lon)
        north_lat = max(ul_lat, lr_lat)
        south_lat = min(ul_lat, lr_lat)

        # Walk across the top and bottom of the image
        for sample in range(0, int(number_of_samples)+1):
            (map_x,
             map_y) = convert_imageXY_to_mapXY(sample, 0.0, ds_transform)
            (top_lon,
             top_lat,
             height) = coord_tf.TransformPoint(map_x, map_y)

            (map_x,
             map_y) = convert_imageXY_to_mapXY(sample, number_of_lines,
                                               ds_transform)
            (bottom_lon,
             bottom_lat,
             height) = coord_tf.TransformPoint(map_x, map_y)

            west_lon = min(top_lon, bottom_lon, west_lon)
            east_lon = max(top_lon, bottom_lon, east_lon)
            north_lat = max(top_lat, bottom_lat, north_lat)
            south_lat = min(top_lat, bottom_lat, south_lat)

        # Walk down the left and right of the image
        for line in range(0, int(number_of_lines)+1):
            (map_x,
             map_y) = convert_imageXY_to_mapXY(0.0, line, ds_transform)
            (left_lon,
             left_lat,
             height) = coord_tf.TransformPoint(map_x, map_y)

            (map_x,
             map_y) = convert_imageXY_to_mapXY(number_of_samples, line,
                                               ds_transform)
            (right_lon,
             right_lat,
             height) = coord_tf.TransformPoint(map_x, map_y)

            west_lon = min(left_lon, right_lon, west_lon)
            east_lon = max(left_lon, right_lon, east_lon)
            north_lat = max(left_lat, right_lat, north_lat)
            south_lat = min(left_lat, right_lat, south_lat)

        # Fix the bounding coordinates if they are outside the valid range,
        # which can happen in antimeridian crossing cases
        if antimeridian_crossing:
            if west_lon < -180:
                west_lon += 360
            if east_lon > 180:
                east_lon -= 360

        # Update the bounding coordinates in the XML
        old_bounding_coordinates = gm.bounding_coordinates
        bounding_coordinates = em.bounding_coordinates()
        bounding_coordinates.west = em.item(west_lon)
        bounding_coordinates.east = em.item(east_lon)
        bounding_coordinates.north = em.item(north_lat)
        bounding_coordinates.south = em.item(south_lat)
        gm.replace(old_bounding_coordinates, bounding_coordinates)

        del ds_transform
        del ds_srs

    except Exception:
        raise


def list_gdal_drivers():
    """Generates a list of all the short names for the GDAL image drivers

    Returns:
        <list>: A list of driver short names
    """

    return [gdal.GetDriver(index).ShortName
            for index in xrange(gdal.GetDriverCount())]


def delete_gdal_drivers(exclusions=list()):
    """Deletes all GDAL image drivers except those in the exclusions list
    """

    for name in list_gdal_drivers():
        if name not in exclusions:
            gdal.GetDriverByName(name).Deregister()


def get_original_projection(img_filename):

    ds = gdal.Open(img_filename)
    if ds is None:
        raise RuntimeError('GDAL failed to open ({})'.format(img_filename))

    ds_srs = osr.SpatialReference()
    ds_srs.ImportFromWkt(ds.GetProjection())

    proj4 = ds_srs.ExportToProj4()

    del ds_srs
    del ds

    return proj4


def build_base_warp_command(args, original_proj4=None):

    # Get the proj4 projection string
    if args.projection == 'none':
        # Default to the provided original proj.4 string
        target_proj4 = original_proj4
    elif args.projection == 'proj4':
        # Use the provided proj.4 projection string for the projection
        target_proj4 = args.proj4_string
    else:
        # Verify and create proj.4 projection string
        target_proj4 = convert_target_projection_to_proj4(args)

    image_extents = build_image_extents_string(args, target_proj4)

    cmd = ['gdalwarp', '-wm', '2048', '-multi', '-of', args.output_format]

    # Subset the image using the specified extents
    if image_extents is not None:
        cmd.extend(['-te'])
        cmd.extend(image_extents)

    # Reproject the data
    if target_proj4 is not None:
        # ***DO NOT*** split the projection string
        # must be quoted with single quotes
        cmd.extend(['-t_srs', "{}".format(target_proj4)])

    return cmd


def main():
    parser = build_command_line_arguments()
    args = parser.parse_args()

    setup_logging(args)

    logger.info('BEGIN - Reprojection Processing')

    verify_command_line(args)

    # We are only supporting ENVI format since this is used by ESPA.
    delete_gdal_drivers(exclusions=['ENVI'])

    # Create an element maker for new items
    em = objectify.ElementMaker(annotate=False, namespace=None, nsmap=None)

    espa_metadata = Metadata(args.xml_filename)
    global_metadata = espa_metadata.xml_object.global_metadata
    bands = espa_metadata.xml_object.bands
    bounding_coordinates = (espa_metadata.xml_object.global_metadata.
                            bounding_coordinates)

    # Default to the pixel information from the metadata
    if args.extent_units and not args.pixel_size_units:
        args.pixel_size = float(bands.band[0].pixel_size.attrib['x'])
        args.pixel_size_units = str(bands.band[0].pixel_size.attrib['units'])

    # Might need this for the base warp command image extents
    original_proj4 = get_original_projection(str(bands.band[0].file_name))

    # Build the base warp command to use
    base_warp_command = \
        build_base_warp_command(args, original_proj4=str(original_proj4))

    # Use the CENTER_LONG gdalwarp configuration setting if using
    # geographic projection and crossing the antimeridian
    if (args.projection == 'lonlat' and
            bounding_coordinates.east < 0 and bounding_coordinates.west > 0):
        base_warp_command.extend(['--config', 'CENTER_LONG', '180'])

    # Process through the bands in the XML file
    for band in bands.band:
        img_filename = str(band.file_name)
        hdr_filename = img_filename.replace('.img', '.hdr')
        aux_filename = img_filename.replace('.img', '.img.aux.xml')
        logger.info("Processing %s" % img_filename)

        resample_method = determine_resample_method(args, band)

        pixel_size = determine_pixel_size(args, global_metadata, band)

        no_data_value = determine_no_data_value(img_filename)

        warped_img_filename = 'warped-%s' % img_filename
        warped_hdr_filename = 'warped-%s' % hdr_filename
        warped_aux_filename = 'warped-%s' % aux_filename

        warp_image(img_filename, warped_img_filename,
                   base_warp_command=base_warp_command,
                   resample_method=resample_method,
                   pixel_size=pixel_size,
                   no_data_value=no_data_value)

        if (band.attrib['category'] == 'image' and
                band.attrib['name'] != 'sr_atmos_opacity'):
            verify_warping(warped_img_filename)

        fix_envi_header(no_data_value, warped_hdr_filename)

        replace_product_files(img_filename, hdr_filename,
                              warped_img_filename, warped_hdr_filename)

        # Remove the *.aux.xml file generated during warp
        if os.path.exists(warped_aux_filename):
            os.unlink(warped_aux_filename)

    # Update the XML to reflect the new warped output
    update_espa_xml(args, espa_metadata)

    # Validate the XML
    espa_metadata.validate()

    # Write it to the XML file
    espa_metadata.write(xml_filename=args.xml_filename)

    del espa_metadata


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if logger is not None:
            logger.exception('Exception Encountered')
            logger.error('END - Reprojection Processing - Failure')
        sys.exit(1)

    if logger is not None:
        logger.info('END - Reprojection Processing - Success')
