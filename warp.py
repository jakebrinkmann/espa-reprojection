
'''
License:
  "NASA Open Source Agreement 1.3"

Description:
  See 'Description' under '__main__' for more details.
  Alters product extents, projections and pixel sizes.

History:
  Original Development (cdr_ecv.py) by David V. Hill, USGS/EROS
  Created Jan/2014 by Ron Dilley, USGS/EROS
    - Gutted the original implementation from cdr_ecv.py and placed it in this
      file.
'''

import os
import sys
import glob
import copy
from cStringIO import StringIO
from argparse import ArgumentParser
from osgeo import gdal, osr
import numpy as np

# espa-common objects and methods
from espa_constants import EXIT_FAILURE
from espa_constants import EXIT_SUCCESS
import metadata_api

# imports from espa_common through processing.__init__.py
from processing import EspaLogging
from processing import settings
from processing import utilities

# local objects and methods
import espa_exception as ee
import parameters


# These contain valid warping options
valid_resample_methods = ['near', 'bilinear', 'cubic', 'cubicspline',
                          'lanczos']
valid_pixel_size_units = ['meters', 'dd']
valid_image_extents_units = ['meters', 'dd']
valid_projections = ['sinu', 'aea', 'utm', 'ps', 'lonlat']
valid_ns = ['north', 'south']
# First entry in the datums is used as the default, it should always be set to
# WGS84
valid_datums = [settings.WGS84, settings.NAD27, settings.NAD83]


# ============================================================================
def build_argument_parser():
    '''
    Description:
      Build the command line argument parser.
    '''

    # Create a command line argument parser
    description = "Alters product extents, projections and pixel sizes"
    parser = ArgumentParser(description=description)

    # Add parameters
    parameters.add_debug_parameter(parser)

    parameters.add_reprojection_parameters(parser,
                                           valid_projections,
                                           valid_ns,
                                           valid_pixel_size_units,
                                           valid_image_extents_units,
                                           valid_resample_methods,
                                           valid_datums)

    parameters.add_work_directory_parameter(parser)

    return parser
# END - build_argument_parser


# ============================================================================
def validate_parameters(parms, scene):
    '''
    Description:
      Make sure all the parameters needed for this and called routines
      is available with the provided input parameters.
    '''

    parameters.validate_reprojection_parameters(parms,
                                                scene,
                                                valid_projections,
                                                valid_ns,
                                                valid_pixel_size_units,
                                                valid_image_extents_units,
                                                valid_resample_methods,
                                                valid_datums)
# END - validate_parameters


# ============================================================================
def build_sinu_proj4_string(central_meridian, false_easting, false_northing):
    '''
    Description:
      Builds a proj.4 string for MODIS
      SR-ORG:6842 Is one of the MODIS spatial reference codes

    Example:
      +proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181
      +ellps=WGS84 +datum=WGS84 +units=m +no_defs
    '''

    proj4_string = ("+proj=sinu +lon_0=%f +x_0=%f +y_0=%f +a=%f +b=%f"
                    " +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
                    % (central_meridian, false_easting, false_northing,
                       settings.SINUSOIDAL_SPHERE_RADIUS,
                       settings.SINUSOIDAL_SPHERE_RADIUS))

    return proj4_string
# END - build_sinu_proj4_string


# ============================================================================
def build_albers_proj4_string(std_parallel_1, std_parallel_2, origin_lat,
                              central_meridian, false_easting, false_northing,
                              datum):
    '''
    Description:
      Builds a proj.4 string for albers equal area

    Example:
      +proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0
      +ellps=GRS80 +datum=NAD83 +units=m +no_defs
    '''

    proj4_string = ("+proj=aea +lat_1=%f +lat_2=%f +lat_0=%f +lon_0=%f"
                    " +x_0=%f +y_0=%f +ellps=GRS80 +datum=%s +units=m"
                    " +no_defs"
                    % (std_parallel_1, std_parallel_2, origin_lat,
                       central_meridian, false_easting, false_northing, datum))

    return proj4_string
# END - build_albers_proj4_string


# ============================================================================
def build_utm_proj4_string(utm_zone, utm_north_south):
    '''
    Description:
      Builds a proj.4 string for utm

    Examples:
      +proj=utm +zone=60 +ellps=WGS84 +datum=WGS84 +units=m +no_defs

      +proj=utm +zone=39 +south +ellps=WGS72 +towgs84=0,0,1.9,0,0,0.814,-0.38
      +units=m +no_defs
    '''
    # TODO - Found this example on the web for south (39), that
    # TODO - specifies the datum instead of "towgs"
    # TODO - gdalsrsinfo EPSG:32739
    # TODO - +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs
    # TODO - It also seems that northern doesn't need the ellipsoid either
    # TODO - gdalsrsinfo EPSG:32660
    # TODO - +proj=utm +zone=60 +datum=WGS84 +units=m +no_defs

    proj4_string = ''
    if str(utm_north_south).lower() == 'north':
        proj4_string = ("+proj=utm +zone=%i +ellps=WGS84 +datum=WGS84"
                        " +units=m +no_defs" % utm_zone)
    elif str(utm_north_south).lower() == 'south':
        proj4_string = ("+proj=utm +zone=%i +south +ellps=WGS72"
                        " +towgs84=0,0,1.9,0,0,0.814,-0.38 +units=m +no_defs"
                        % utm_zone)
    else:
        raise ValueError("Invalid utm_north_south argument[%s]"
                         " Argument must be one of 'north' or 'south'"
                         % utm_north_south)

    return proj4_string
# END - build_utm_proj4_string


# ============================================================================
def build_ps_proj4_string(lat_ts, lon_pole, origin_lat,
                          false_easting, false_northing):
    '''
    Description:
      Builds a proj.4 string for polar stereographic
      gdalsrsinfo 'EPSG:3031'

    Examples:
      +proj=stere +lat_0=90 +lat_ts=71 +lon_0=0 +k=1 +x_0=0 +y_0=0
        +datum=WGS84 +units=m +no_defs

      +proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0
        +datum=WGS84 +units=m +no_defs
    '''

    proj4_string = ("+proj=stere +lat_ts=%f +lat_0=%f +lon_0=%f +k_0=1.0"
                    " +x_0=%f +y_0=%f +datum=WGS84 +units=m +no_defs"
                    % (lat_ts, origin_lat, lon_pole,
                       false_easting, false_northing))

    return proj4_string
# END - build_ps_proj4_string


# ============================================================================
def convert_target_projection_to_proj4(parms):
    '''
    Description:
      Checks to see if the reproject parameter was set.  If set the
      target projection is validated against the implemented projections and
      depending on the projection, the correct proj4 parameters are returned.
    '''

    projection = None
    target_projection = None

    target_projection = parms['target_projection']

    if target_projection == "sinu":
        projection = \
            build_sinu_proj4_string(parms['central_meridian'],
                                    parms['false_easting'],
                                    parms['false_northing'])

    elif target_projection == "aea":
        projection = \
            build_albers_proj4_string(parms['std_parallel_1'],
                                      parms['std_parallel_2'],
                                      parms['origin_lat'],
                                      parms['central_meridian'],
                                      parms['false_easting'],
                                      parms['false_northing'],
                                      parms['datum'])

    elif target_projection == "utm":
        projection = \
            build_utm_proj4_string(parms['utm_zone'],
                                   parms['utm_north_south'])

    elif target_projection == "ps":
        projection = build_ps_proj4_string(parms['latitude_true_scale'],
                                           parms['longitude_pole'],
                                           parms['origin_lat'],
                                           parms['false_easting'],
                                           parms['false_northing'])

    elif target_projection == "lonlat":
        projection = settings.GEOGRAPHIC_PROJ4_STRING

    return str(projection)
# END - convert_target_projection_to_proj4


# ============================================================================
def projection_minbox(ul_lon, ul_lat, lr_lon, lr_lat,
                      target_proj4, pixel_size, pixel_size_units):
    '''
    Description:
      Determines the minimum box in map coordinates that contains the
      geographic coordinates.  Minimum and maximum extent values are returned
      in map coordinates.

    Parameters:
      ul_lon       = Upper Left longitude in decimal degrees
      ul_lat       = Upper Left latitude in decimal degrees
      lr_lon       = Lower Right longitude in decimal degrees
      lr_lat       = Lower Right latitude in decimal degrees
      target_proj4 = The user supplied target proj4 string
      pixel_size   = The target pixel size in meters used to step along the
                     projected area boundary
      pixel_size_units = The units the pixel size is in 'dd' or 'meters'

    Returns:
        (min_x, min_y, max_x, max_y) in meters
    '''

    logger = EspaLogging.get_logger(settings.PROCESSING_LOGGER)

    logger.info("Determining Image Extents For Requested Projection")

    # We are always going to be geographic
    source_proj4 = settings.GEOGRAPHIC_PROJ4_STRING

    logger.info("Using source projection [%s]" % source_proj4)
    logger.info("Using target projection [%s]" % target_proj4)

    # Create and initialize the source SRS
    source_srs = osr.SpatialReference()
    source_srs.ImportFromProj4(source_proj4)

    # Create and initialize the target SRS
    target_srs = osr.SpatialReference()
    target_srs.ImportFromProj4(target_proj4)

    # Create the transformation object
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    # Determine the step in decimal degrees
    step = pixel_size
    if pixel_size_units == 'meters':
        # Convert it to decimal degrees
        step = settings.DEG_FOR_1_METER * pixel_size

    # Determine the lat and lon values to iterate over
    longitudes = np.arange(ul_lon, lr_lon, step, np.float)
    latitudes = np.arange(lr_lat, ul_lat, step, np.float)

    # Initialization using the two corners
    (ul_x, ul_y, z) = transform.TransformPoint(ul_lon, ul_lat)
    (lr_x, lr_y, z) = transform.TransformPoint(lr_lon, lr_lat)

    min_x = min(ul_x, lr_x)
    max_x = max(ul_x, lr_x)
    min_y = min(ul_y, lr_y)
    max_y = max(ul_y, lr_y)

    logger.info('Direct translation of the provided geographic coordinates')
    logger.info(','.join(['min_x', 'min_y', 'max_x', 'max_y']))
    logger.info(','.join([str(min_x), str(min_y), str(max_x), str(max_y)]))

    # Walk across the top and bottom of the geographic coordinates
    for lon in longitudes:
        # Upper side
        (ux, uy, z) = transform.TransformPoint(lon, ul_lat)

        # Lower side
        (lx, ly, z) = transform.TransformPoint(lon, lr_lat)

        min_x = min(ux, lx, min_x)
        max_x = max(ux, lx, max_x)
        min_y = min(uy, ly, min_y)
        max_y = max(uy, ly, max_y)

    # Walk along the left and right of the geographic coordinates
    for lat in latitudes:
        # Left side
        (lx, ly, z) = transform.TransformPoint(ul_lon, lat)

        # Right side
        (rx, ry, z) = transform.TransformPoint(lr_lon, lat)

        min_x = min(rx, lx, min_x)
        max_x = max(rx, lx, max_x)
        min_y = min(ry, ly, min_y)
        max_y = max(ry, ly, max_y)

    del(transform)
    del(source_srs)
    del(target_srs)

    logger.info('Map coordinates after minbox determination')
    logger.info(','.join(['min_x', 'min_y', 'max_x', 'max_y']))
    logger.info(','.join([str(min_x), str(min_y), str(max_x), str(max_y)]))

    return (min_x, min_y, max_x, max_y)
# END - projection_minbox


# ============================================================================
def build_image_extents_string(parms, target_proj4):
    '''
    Description:
      Build the gdal_warp image extents string from the determined min and max
      values.

    Returns:
        str('min_x min_y max_x max_y')
    '''

    # Nothing to do if we are not sub-setting the data
    if not parms['image_extents']:
        return None

    target_projection = parms['target_projection']

    # Get the image extents string
    if (parms['image_extents_units'] == 'dd'
            and (target_projection is None or target_projection != 'lonlat')):

        (min_x, min_y, max_x, max_y) = \
            projection_minbox(parms['minx'], parms['maxy'],
                              parms['maxx'], parms['miny'],
                              target_proj4,
                              parms['pixel_size'],
                              parms['pixel_size_units'])
    else:
        (min_x, min_y, max_x, max_y) = (parms['minx'], parms['miny'],
                                        parms['maxx'], parms['maxy'])

    return ' '.join([str(min_x), str(min_y), str(max_x), str(max_y)])
# END - build_image_extents_string


# ============================================================================
def build_base_warp_command(parms, output_format='envi', original_proj4=None):

    # Get the proj4 projection string
    if parms['projection'] is not None:
        # Use the provided proj.4 projection string for the projection
        target_proj4 = parms['projection']
    elif parms['reproject']:
        # Verify and create proj.4 projection string
        target_proj4 = convert_target_projection_to_proj4(parms)
    else:
        # Default to the provided original proj.4 string
        target_proj4 = original_proj4

    image_extents = build_image_extents_string(parms, target_proj4)

    cmd = ['gdalwarp', '-wm', '2048', '-multi', '-of', output_format]

    # Subset the image using the specified extents
    if image_extents is not None:
        cmd.extend(['-te', image_extents])

    # Reproject the data
    if target_proj4 is not None:
        # ***DO NOT*** split the projection string
        # must be quoted with single quotes
        cmd.extend(['-t_srs', "'%s'" % target_proj4])

    return cmd
# END - build_base_warp_command


# ============================================================================
def warp_image(source_file, output_file,
               base_warp_command=None,
               resample_method='near',
               pixel_size=None,
               no_data_value=None):
    '''
    Description:
      Executes the warping command on the specified source file
    '''

    logger = EspaLogging.get_logger(settings.PROCESSING_LOGGER)

    try:
        # Turn GDAL PAM off to prevent *.aux.xml files
        os.environ['GDAL_PAM_ENABLED'] = 'NO'

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

        cmd = ' '.join(cmd)
        logger.info("Warping %s with %s" % (source_file, cmd))

        output = utilities.execute_cmd(cmd)
        if len(output) > 0:
            logger.info(output)

    except Exception:
        raise

    finally:
        # Remove the environment variable we set above
        del os.environ['GDAL_PAM_ENABLED']
# END - warp_image


# ============================================================================
def convert_imageXY_to_mapXY(image_x, image_y, transform):
    '''
    Description:
      Translate image coordinates into mapp coordinates
    '''

    map_x = transform[0] + image_x * transform[1] + image_y * transform[2]
    map_y = transform[3] + image_x * transform[4] + image_y * transform[5]

    return (map_x, map_y)
# END - convert_imageXY_to_mapXY


# ============================================================================
def update_espa_xml(parms, xml, xml_filename):

    logger = EspaLogging.get_logger(settings.PROCESSING_LOGGER)

    try:
        # Default the datum to WGS84
        datum = settings.WGS84
        if parms['datum'] is not None:
            datum = parms['datum']

        bands = xml.get_bands()
        for band in bands.band:
            img_filename = band.get_file_name()
            logger.info("Updating XML for %s" % img_filename)

            ds = gdal.Open(img_filename)
            if ds is None:
                msg = "GDAL failed to open (%s)" % img_filename
                raise RuntimeError(msg)

            try:
                ds_band = ds.GetRasterBand(1)
                ds_transform = ds.GetGeoTransform()
                ds_srs = osr.SpatialReference()
                ds_srs.ImportFromWkt(ds.GetProjection())
            except Exception, e:
                raise ee.ESPAException(ee.ErrorCodes.warping,
                                       str(e)), None, sys.exc_info()[2]

            projection_name = ds_srs.GetAttrValue('PROJECTION')

            number_of_lines = float(ds_band.YSize)
            number_of_samples = float(ds_band.XSize)
            # Need to abs these because they are coming from the transform,
            # which may becorrect for the transform,
            # but not how us humans understand it
            x_pixel_size = abs(ds_transform[1])
            y_pixel_size = abs(ds_transform[5])

            del (ds_band)
            del (ds)

            # Update the band information in the XML file
            band.set_nlines(number_of_lines)
            band.set_nsamps(number_of_samples)
            band_pixel_size = band.get_pixel_size()
            band_pixel_size.set_x(x_pixel_size)
            band_pixel_size.set_y(y_pixel_size)

            # For sanity report the resample method applied to the data
            resample_method = band.get_resample_method()
            logger.info("RESAMPLE METHOD [%s]" % resample_method)

            # We only support one unit type for each projection
            if projection_name is not None:
                if projection_name.lower().startswith('transverse_mercator'):
                    band_pixel_size.set_units('meters')
                elif projection_name.lower().startswith('polar'):
                    band_pixel_size.set_units('meters')
                elif projection_name.lower().startswith('albers'):
                    band_pixel_size.set_units('meters')
                elif projection_name.lower().startswith('sinusoidal'):
                    band_pixel_size.set_units('meters')
            else:
                # Must be Geographic Projection
                band_pixel_size.set_units('degrees')

        ######################################################################
        # Fix the projection information for the warped data
        ######################################################################
        gm = xml.get_global_metadata()

        # If the image extents were changed, then the scene center time is
        # meaningless so just remove it
        # We don't have any way to calculate a new one
        if parms['image_extents']:
            del gm.scene_center_time
            gm.scene_center_time = None

        # Remove the projection parameter object from the structure so that it
        # can be replaced with the new one
        # Geographic doesn't have one
        if gm.projection_information.utm_proj_params is not None:
            del gm.projection_information.utm_proj_params
            gm.projection_information.utm_proj_params = None

        if gm.projection_information.ps_proj_params is not None:
            del gm.projection_information.ps_proj_params
            gm.projection_information.ps_proj_params = None

        if gm.projection_information.albers_proj_params is not None:
            del gm.projection_information.albers_proj_params
            gm.projection_information.albers_proj_params = None

        if gm.projection_information.sin_proj_params is not None:
            del gm.projection_information.sin_proj_params
            gm.projection_information.sin_proj_params = None

        # Rebuild the projection parameters
        projection_name = ds_srs.GetAttrValue('PROJECTION')
        if projection_name is not None:
            # ----------------------------------------------------------------
            if projection_name.lower().startswith('transverse_mercator'):
                logger.info("---- Updating UTM Parameters")
                # Get the parameter values
                zone = int(ds_srs.GetUTMZone())
                # Get a new UTM projection parameter object and populate it
                utm_projection = metadata_api.utm_proj_params()
                utm_projection.set_zone_code(zone)
                # Add the object to the projection information
                gm.projection_information.set_utm_proj_params(utm_projection)
                # Update the attribute values
                gm.projection_information.set_projection("UTM")
                gm.projection_information.set_datum(settings.WGS84)
            # ----------------------------------------------------------------
            elif projection_name.lower().startswith('polar'):
                logger.info("---- Updating Polar Stereographic Parameters")
                # Get the parameter values
                latitude_true_scale = ds_srs.GetProjParm('latitude_of_origin')
                longitude_pole = ds_srs.GetProjParm('central_meridian')
                false_easting = ds_srs.GetProjParm('false_easting')
                false_northing = ds_srs.GetProjParm('false_northing')
                # Get a new PS projection parameter object and populate it
                ps_projection = metadata_api.ps_proj_params()
                ps_projection.set_latitude_true_scale(latitude_true_scale)
                ps_projection.set_longitude_pole(longitude_pole)
                ps_projection.set_false_easting(false_easting)
                ps_projection.set_false_northing(false_northing)
                # Add the object to the projection information
                gm.projection_information.set_ps_proj_params(ps_projection)
                # Update the attribute values
                gm.projection_information.set_projection("PS")
                gm.projection_information.set_datum(settings.WGS84)
            # ----------------------------------------------------------------
            elif projection_name.lower().startswith('albers'):
                logger.info("---- Updating Albers Equal Area Parameters")
                # Get the parameter values
                standard_parallel1 = ds_srs.GetProjParm('standard_parallel_1')
                standard_parallel2 = ds_srs.GetProjParm('standard_parallel_2')
                origin_latitude = ds_srs.GetProjParm('latitude_of_center')
                central_meridian = ds_srs.GetProjParm('longitude_of_center')
                false_easting = ds_srs.GetProjParm('false_easting')
                false_northing = ds_srs.GetProjParm('false_northing')
                # Get a new ALBERS projection parameter object and populate it
                albers_projection = metadata_api.albers_proj_params()
                albers_projection.set_standard_parallel1(standard_parallel1)
                albers_projection.set_standard_parallel2(standard_parallel2)
                albers_projection.set_origin_latitude(origin_latitude)
                albers_projection.set_central_meridian(central_meridian)
                albers_projection.set_false_easting(false_easting)
                albers_projection.set_false_northing(false_northing)
                # Add the object to the projection information
                gm.projection_information. \
                    set_albers_proj_params(albers_projection)
                # Update the attribute values
                gm.projection_information.set_projection("ALBERS")
                # This projection can have different datums, so use the datum
                # requested by the user
                gm.projection_information.set_datum(datum)
            # ----------------------------------------------------------------
            elif projection_name.lower().startswith('sinusoidal'):
                logger.info("---- Updating Sinusoidal Parameters")
                # Get the parameter values
                central_meridian = ds_srs.GetProjParm('longitude_of_center')
                false_easting = ds_srs.GetProjParm('false_easting')
                false_northing = ds_srs.GetProjParm('false_northing')
                # Get a new SIN projection parameter object and populate it
                sin_projection = metadata_api.sin_proj_params()
                sin_projection.set_sphere_radius(
                    settings.SINUSOIDAL_SPHERE_RADIUS)
                sin_projection.set_central_meridian(central_meridian)
                sin_projection.set_false_easting(false_easting)
                sin_projection.set_false_northing(false_northing)
                # Add the object to the projection information
                gm.projection_information.set_sin_proj_params(sin_projection)
                # Update the attribute values
                gm.projection_information.set_projection("SIN")
                # This projection doesn't have a datum
                del gm.projection_information.datum
                gm.projection_information.datum = None
        else:
            # ----------------------------------------------------------------
            # Must be Geographic Projection
            logger.info("---- Updating Geographic Parameters")
            gm.projection_information.set_projection('GEO')
            gm.projection_information.set_datum(settings.WGS84)  # WGS84 only
            gm.projection_information.set_units('degrees')  # always degrees

        # Fix the UL and LR center of pixel map coordinates
        (map_ul_x, map_ul_y) = convert_imageXY_to_mapXY(0.5, 0.5,
                                                        ds_transform)
        (map_lr_x, map_lr_y) = convert_imageXY_to_mapXY(
            number_of_samples - 0.5, number_of_lines - 0.5, ds_transform)
        for cp in gm.projection_information.corner_point:
            if cp.location == 'UL':
                cp.set_x(map_ul_x)
                cp.set_y(map_ul_y)
            if cp.location == 'LR':
                cp.set_x(map_lr_x)
                cp.set_y(map_lr_y)

        # Fix the UL and LR center of pixel latitude and longitude coordinates
        srs_lat_lon = ds_srs.CloneGeogCS()
        coord_tf = osr.CoordinateTransformation(ds_srs, srs_lat_lon)
        for corner in gm.corner:
            if corner.location == 'UL':
                (lon, lat, height) = \
                    coord_tf.TransformPoint(map_ul_x, map_ul_y)
                corner.set_longitude(lon)
                corner.set_latitude(lat)
            if corner.location == 'LR':
                (lon, lat, height) = \
                    coord_tf.TransformPoint(map_lr_x, map_lr_y)
                corner.set_longitude(lon)
                corner.set_latitude(lat)

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
            (map_x, map_y) = \
                convert_imageXY_to_mapXY(sample, 0.0, ds_transform)
            (top_lon, top_lat, height) = coord_tf.TransformPoint(map_x, map_y)

            (map_x, map_y) = \
                convert_imageXY_to_mapXY(sample, number_of_lines, ds_transform)
            (bottom_lon, bottom_lat, height) = \
                coord_tf.TransformPoint(map_x, map_y)

            west_lon = min(top_lon, bottom_lon, west_lon)
            east_lon = max(top_lon, bottom_lon, east_lon)
            north_lat = max(top_lat, bottom_lat, north_lat)
            south_lat = min(top_lat, bottom_lat, south_lat)

        # Walk down the left and right of the image
        for line in range(0, int(number_of_lines)+1):
            (map_x, map_y) = \
                convert_imageXY_to_mapXY(0.0, line, ds_transform)
            (left_lon, left_lat, height) = \
                coord_tf.TransformPoint(map_x, map_y)

            (map_x, map_y) = \
                convert_imageXY_to_mapXY(number_of_samples, line, ds_transform)
            (right_lon, right_lat, height) = \
                coord_tf.TransformPoint(map_x, map_y)

            west_lon = min(left_lon, right_lon, west_lon)
            east_lon = max(left_lon, right_lon, east_lon)
            north_lat = max(left_lat, right_lat, north_lat)
            south_lat = min(left_lat, right_lat, south_lat)

        # Update the bounding coordinates in the XML
        bounding_coords = gm.get_bounding_coordinates()
        bounding_coords.set_west(west_lon)
        bounding_coords.set_east(east_lon)
        bounding_coords.set_north(north_lat)
        bounding_coords.set_south(south_lat)

        del (ds_transform)
        del (ds_srs)

        # Write out a new XML file after validation
        logger.info("---- Validating XML Modifications and"
                    " Creating Temp Output File")
        tmp_xml_filename = 'tmp-%s' % xml_filename
        with open(tmp_xml_filename, 'w') as tmp_fd:
            # Call the export with validation
            metadata_api.export(tmp_fd, xml)

        # Remove the original
        if os.path.exists(xml_filename):
            os.unlink(xml_filename)

        # Rename the temp file back to the original name
        os.rename(tmp_xml_filename, xml_filename)

    except Exception, e:
        raise ee.ESPAException(ee.ErrorCodes.warping,
                               str(e)), None, sys.exc_info()[2]
# END - update_espa_xml


# ============================================================================
def get_original_projection(img_filename):

    ds = gdal.Open(img_filename)
    if ds is None:
        raise RuntimeError("GDAL failed to open (%s)" % img_filename)

    ds_srs = osr.SpatialReference()
    ds_srs.ImportFromWkt(ds.GetProjection())

    proj4 = ds_srs.ExportToProj4()

    del (ds_srs)
    del (ds)

    return proj4
# END - get_original_projection


# ============================================================================
def warp_espa_data(parms, scene, xml_filename=None):
    '''
    Description:
      Warp each espa science product to the parameters specified in the parms
    '''

    logger = EspaLogging.get_logger(settings.PROCESSING_LOGGER)

    # Validate the parameters
    validate_parameters(parms, scene)
    logger.debug(parms)

    # ------------------------------------------------------------------------
    # De-register the DOQ drivers since they may cause a problem with some of
    # our generated imagery.  And we are only processing envi format today
    # inside the processing code.
    doq1 = gdal.GetDriverByName('DOQ1')
    doq2 = gdal.GetDriverByName('DOQ2')
    doq1.Deregister()
    doq2.Deregister()
    # ------------------------------------------------------------------------

    # Verify something was provided for the XML filename
    if xml_filename is None or xml_filename == '':
        raise ee.ESPAException(ee.ErrorCodes.warping, "Missing XML Filename")

    # Change to the working directory
    current_directory = os.getcwd()
    os.chdir(parms['work_directory'])

    try:
        xml = metadata_api.parse(xml_filename, silence=True)
        bands = xml.get_bands()
        global_metadata = xml.get_global_metadata()
        satellite = global_metadata.get_satellite()

        # Might need this for the base warp command image extents
        original_proj4 = get_original_projection(bands.band[0].get_file_name())

        # Build the base warp command to use
        base_warp_command = \
            build_base_warp_command(parms, original_proj4=str(original_proj4))

        # Determine the user specified resample method
        user_resample_method = 'near'  # default
        if parms['resample_method'] is not None:
            user_resample_method = parms['resample_method']

        # Process through the bands in the XML file
        for band in bands.band:
            img_filename = band.get_file_name()
            hdr_filename = img_filename.replace('.img', '.hdr')
            logger.info("Processing %s" % img_filename)

            # Reset the resample method to the user specified value
            resample_method = user_resample_method

            # Always use near for qa bands
            category = band.get_category()
            if category == 'qa':
                resample_method = 'near'  # over-ride with 'near'

            # Update the XML metadata object for the resampling method used
            # Later update_espa_xml is used to update the XML file
            if resample_method == 'near':
                band.set_resample_method('nearest neighbor')
            if resample_method == 'bilinear':
                band.set_resample_method('bilinear')
            if resample_method == 'cubic':
                band.set_resample_method('cubic convolution')

            # Figure out the pixel size to use
            pixel_size = parms['pixel_size']

            # EXECUTIVE DECISION(Calli) - ESPA Issue 185
            #    - If the band is Landsat 7 Band 8 do not resize the pixels.
            if satellite == 'LANDSAT_7' and band.get_name() == 'band8':
                if parms['target_projection'] == 'lonlat':
                    pixel_size = settings.DEG_FOR_15_METERS
                else:
                    pixel_size = float(band.pixel_size.x)

            # Open the image to read the no data value out since the internal
            # ENVI driver for GDAL does not output it, even if it is known
            ds = gdal.Open(img_filename)
            if ds is None:
                raise RuntimeError("GDAL failed to open (%s)" % img_filename)

            ds_band = None
            try:
                ds_band = ds.GetRasterBand(1)
            except Exception, e:
                raise ee.ESPAException(ee.ErrorCodes.warping,
                                       str(e)), None, sys.exc_info()[2]

            # Save the no data value since gdalwarp does not write it out when
            # using the ENVI format
            no_data_value = ds_band.GetNoDataValue()
            if no_data_value is not None:
                # TODO - We don't process any floating point data types.  Yet
                # Convert to an integer then string
                no_data_value = str(int(no_data_value))

            # Force a freeing of the memory
            del (ds_band)
            del (ds)

            tmp_img_filename = 'tmp-%s' % img_filename
            tmp_hdr_filename = 'tmp-%s' % hdr_filename

            warp_image(img_filename, tmp_img_filename,
                       base_warp_command=base_warp_command,
                       resample_method=resample_method,
                       pixel_size=pixel_size,
                       no_data_value=no_data_value)

            ##################################################################
            ##################################################################
            # Get new everything for the re-projected band
            ##################################################################
            ##################################################################

            # Update the tmp ENVI header with our own values for some fields
            sb = StringIO()
            with open(tmp_hdr_filename, 'r') as tmp_fd:
                while True:
                    line = tmp_fd.readline()
                    if not line:
                        break
                    if (line.startswith('data ignore value')
                            or line.startswith('description')):
                        pass
                    else:
                        sb.write(line)

                    if line.startswith('description'):
                        # This may be on multiple lines so read lines until
                        # we find the closing brace
                        if not line.strip().endswith('}'):
                            while 1:
                                next_line = tmp_fd.readline()
                                if (not next_line
                                        or next_line.strip().endswith('}')):
                                    break
                        sb.write('description = {ESPA-generated file}\n')
                    elif (line.startswith('data type')
                          and (no_data_value is not None)):
                        sb.write('data ignore value = %s\n' % no_data_value)
            # END - with tmp_fd

            # Do the actual replace here
            with open(tmp_hdr_filename, 'w') as tmp_fd:
                tmp_fd.write(sb.getvalue())

            # Remove the original files, they are replaced in following code
            if os.path.exists(img_filename):
                os.unlink(img_filename)
            if os.path.exists(hdr_filename):
                os.unlink(hdr_filename)

            # Rename the temps file back to the original name
            os.rename(tmp_img_filename, img_filename)
            os.rename(tmp_hdr_filename, hdr_filename)
        # END for each band in the XML file

        # Update the XML to reflect the new warped output
        update_espa_xml(parms, xml, xml_filename)

        del (xml)

    except Exception, e:
        raise ee.ESPAException(ee.ErrorCodes.warping,
                               str(e)), None, sys.exc_info()[2]
    finally:
        # Change back to the previous directory
        os.chdir(current_directory)
# END - warp_espa_data


# ============================================================================
def reformat(metadata_filename, work_directory, input_format, output_format):
    '''
    Description:
      Re-format the bands to the specified format using our raw binary tools
      or gdal, whichever is appropriate for the task.

      Input espa:
          Output Formats: envi(espa), gtiff, and hdf
    '''

    logger = EspaLogging.get_logger(settings.PROCESSING_LOGGER)

    # Don't do anything if they match
    if input_format == output_format:
        return

    # Change to the working directory
    current_directory = os.getcwd()
    os.chdir(work_directory)

    try:
        # Convert from our internal ESPA/ENVI format to GeoTIFF
        if input_format == 'envi' and output_format == 'gtiff':
            gtiff_name = metadata_filename.rstrip('.xml')
            # Call with deletion of source files
            cmd = ' '.join(['convert_espa_to_gtif', '--del_src_files',
                            '--xml', metadata_filename,
                            '--gtif', gtiff_name])

            output = ''
            try:
                output = utilities.execute_cmd(cmd)

                # Rename the XML file back to *.xml from *_gtif.xml
                meta_gtiff_name = metadata_filename.split('.xml')[0]
                meta_gtiff_name = ''.join([meta_gtiff_name, '_gtif.xml'])

                os.rename(meta_gtiff_name, metadata_filename)
            except Exception, e:
                raise ee.ESPAException(ee.ErrorCodes.reformat,
                                       str(e)), None, sys.exc_info()[2]
            finally:
                if len(output) > 0:
                    logger.info(output)

            # Remove all the *.tfw files since gtiff was chosen a bunch may
            # be present
            files_to_remove = glob.glob('*.tfw')
            if len(files_to_remove) > 0:
                cmd = ' '.join(['rm', '-rf'] + files_to_remove)
                logger.info(' '.join(['REMOVING TFW DATA COMMAND:', cmd]))

                output = ''
                try:
                    output = utilities.execute_cmd(cmd)
                except Exception, e:
                    raise ee.ESPAException(ee.ErrorCodes.reformat,
                                           str(e)), None, sys.exc_info()[2]
                finally:
                    if len(output) > 0:
                        logger.info(output)

        # Convert from our internal ESPA/ENVI format to HDF
        elif input_format == 'envi' and output_format == 'hdf-eos2':
            # convert_espa_to_hdf
            hdf_name = metadata_filename.replace('.xml', '.hdf')
            # Call with deletion of source files
            cmd = ' '.join(['convert_espa_to_hdf', '--del_src_files',
                            '--xml', metadata_filename,
                            '--hdf', hdf_name])

            output = ''
            try:
                output = utilities.execute_cmd(cmd)

                # Rename the XML file back to *.xml from *_hdf.xml
                meta_hdf_name = metadata_filename.replace('.xml', '_hdf.xml')

                os.rename(meta_hdf_name, metadata_filename)
            except Exception, e:
                raise ee.ESPAException(ee.ErrorCodes.reformat,
                                       str(e)), None, sys.exc_info()[2]
            finally:
                if len(output) > 0:
                    logger.info(output)

        # Requested conversion not implemented
        else:
            raise ValueError("Unsupported reformat combination (%s, %s)"
                             % (input_format, output_format))

    except Exception, e:
        raise ee.ESPAException(ee.ErrorCodes.reformat,
                               str(e)), None, sys.exc_info()[2]
    finally:
        # Change back to the previous directory
        os.chdir(current_directory)
# END - reformat


# ============================================================================
if __name__ == '__main__':
    '''
    Description:
      Read parameters from the command line and build a JSON dictionary from
      them.  Pass the JSON dictionary to the process routine.
    '''

    # Build the command line argument parser
    parser = build_argument_parser()

    # Parse the command line arguments
    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    # Configure logging
    EspaLogging.configure(settings.PROCESSING_LOGGER, order='test',
                          product='product', debug=args.debug)
    logger = EspaLogging.get_logger(settings.PROCESSING_LOGGER)

    # Build our JSON formatted input from the command line parameters
    options = {k: args_dict[k] for k in args_dict if args_dict[k] is not None}

    try:
        # Call the main processing routine
        warp_espa_data(options, parms['scene'])
    except Exception, e:
        if hasattr(e, 'output'):
            logger.error("Output [%s]" % e.output)
        logger.exception("Processing failed")
        sys.exit(EXIT_FAILURE)

    sys.exit(EXIT_SUCCESS)
