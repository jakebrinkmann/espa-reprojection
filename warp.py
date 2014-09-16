#! /usr/bin/env python

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
from cStringIO import StringIO
from argparse import ArgumentParser
from osgeo import gdal, osr

# espa-common objects and methods
from espa_constants import *
import metadata_api

# imports from espa/espa_common
try:
    from espa_logging import EspaLogging
except:
    from espa_common.espa_logging import EspaLogging

try:
    import utilities
except:
    from espa_common import utilities


# local objects and methods
import espa_exception as ee
import parameters
import util


# We are only supporting one radius
SINUSOIDAL_SPHERE_RADIUS = 6371007.181

# Some defines for common pixels sizes in decimal degrees
DEG_FOR_30_METERS = 0.0002695
DEG_FOR_15_METERS = DEG_FOR_30_METERS / 2.0

# Supported datums - the strings for them
WGS84 = 'WGS84'
NAD27 = 'NAD27'
NAD83 = 'NAD83'

# These contain valid warping options
valid_resample_methods = ['near', 'bilinear', 'cubic', 'cubicspline',
                          'lanczos']
valid_pixel_size_units = ['meters', 'dd']
valid_image_extents_units = ['meters', 'dd']
valid_projections = ['sinu', 'aea', 'utm', 'ps', 'lonlat']
valid_ns = ['north', 'south']
# First entry in the datums is used as the default, it should always be set to
# WGS84
valid_datums = [WGS84, NAD27, NAD83]


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

    global SINUSOIDAL_SPHERE_RADIUS

    proj4_string = ("'+proj=sinu +lon_0=%f +x_0=%f +y_0=%f +a=%f +b=%f"
                    " +ellps=WGS84 +datum=WGS84 +units=m +no_defs'"
                    % (central_meridian, false_easting, false_northing,
                       SINUSOIDAL_SPHERE_RADIUS, SINUSOIDAL_SPHERE_RADIUS))

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

    proj4_string = ("'+proj=aea +lat_1=%f +lat_2=%f +lat_0=%f +lon_0=%f"
                    " +x_0=%f +y_0=%f +ellps=GRS80 +datum=%s +units=m"
                    " +no_defs'"
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
        proj4_string = ("'+proj=utm +zone=%i +ellps=WGS84 +datum=WGS84"
                        " +units=m +no_defs'" % utm_zone)
    elif str(utm_north_south).lower() == 'south':
        proj4_string = ("'+proj=utm +zone=%i +south +ellps=WGS72"
                        " +towgs84=0,0,1.9,0,0,0.814,-0.38 +units=m +no_defs'"
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

    proj4_string = ("'+proj=stere +lat_ts=%f +lat_0=%f +lon_0=%f +k_0=1.0"
                    " +x_0=%f +y_0=%f +datum=WGS84 +units=m +no_defs'"
                    % (lat_ts, origin_lat, lon_pole,
                       false_easting, false_northing))

    return proj4_string
# END - build_ps_proj4_string


# ============================================================================
def build_geographic_proj4_string():
    '''
    Description:
      Builds a proj.4 string for geographic
      gdalsrsinfo 'EPSG:4326'

    Example:
        +proj=longlat +datum=WGS84 +no_defs

    '''

    return "'+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'"
# END - build_geographic_proj4_string


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
        projection = build_geographic_proj4_string()

    return projection
# END - convert_target_projection_to_proj4


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
def build_warp_command(source_file, output_file, output_format='envi',
                       min_x=None, min_y=None, max_x=None, max_y=None,
                       pixel_size=None, projection=None,
                       resample_method=None, no_data_value=None):
    '''
    Description:
      Builds the GDAL warp command to convert the data
    '''

    logger = EspaLogging.get_logger('espa.processing')

    cmd = ['gdalwarp', '-wm', '2048', '-multi', '-of', output_format]

    # Subset the image using the specified extents
    if ((min_x is not None) and (min_y is not None)
            and (max_x is not None) and (max_y is not None)):

        logger.debug("Image Extents: %f, %f, %f, %f"
                     % (min_x, min_y, max_x, max_y))
        cmd.extend(['-te', str(min_x), str(min_y), str(max_x), str(max_y)])

    # Resize the pixels
    if pixel_size is not None:
        cmd.extend(['-tr', str(pixel_size), str(pixel_size)])

    # Reproject the data
    if projection is not None:
        # ***DO NOT*** split the projection string
        cmd.extend(['-t_srs', projection])

    # Specify the resampling method
    if resample_method is not None:
        cmd.extend(['-r', resample_method])

    if no_data_value is not None:
        cmd.extend(['-srcnodata', no_data_value])
        cmd.extend(['-dstnodata', no_data_value])

    cmd.extend([source_file, output_file])

    return cmd
# END - build_warp_command


# ============================================================================
def parse_hdf_subdatasets(hdf_file):
    '''
    Description:
      Finds all the subdataset names in an hdf file
    '''

    cmd = ' '.join(['gdalinfo', hdf_file])
    output = utilities.execute_cmd(cmd)
    name = ''
    description = ''
    for line in output.split('\n'):
        line_lower = line.strip().lower()

        # logic heavily based on the output order from gdalinfo
        if (line_lower.startswith('subdataset')
                and line_lower.find('_name') != -1):

            parts = line.split('=')
            name = parts[1]

        if (line_lower.startswith('subdataset')
                and line_lower.find('_desc') != -1):

            parts = line.split('=')
            description = parts[1]
            yield (description, name)
# END - parse_hdf_subdatasets


# ============================================================================
def get_no_data_value(filename):
    '''
    Description:
      Returns the 'nodata value' associated with a georeferenced image.
    '''

    cmd = ' '.join(['gdalinfo', filename])
    output = utilities.execute_cmd(cmd)

    no_data_value = None
    for line in output.split('\n'):
        line_lower = line.strip().lower()

        if line_lower.startswith('nodata value'):
            no_data_value = line_lower.split('=')[1]  # take second element

    return no_data_value
# END - get_no_data_value


# ============================================================================
def run_warp(source_file, output_file, output_format='envi',
             min_x=None, min_y=None, max_x=None, max_y=None,
             pixel_size=None, projection=None,
             resample_method=None, no_data_value=None):
    '''
    Description:
      Executes the warping command on the specified source file
    '''

    logger = EspaLogging.get_logger('espa.processing')

    try:
        # Turn GDAL PAM off to prevent *.aux.xml files
        os.environ['GDAL_PAM_ENABLED'] = 'NO'

        cmd = build_warp_command(source_file, output_file, output_format,
                                 min_x, min_y, max_x, max_y, pixel_size,
                                 projection, resample_method, no_data_value)
        logger.debug(cmd)
        cmd = ' '.join(cmd)

        logger.info("Warping %s with %s" % (source_file, cmd))
        output = utilities.execute_cmd(cmd)
        if len(output) > 0:
            logger.info(output)

    except Exception, e:
        raise

    finally:
        # Remove the environment variable we set above
        del os.environ['GDAL_PAM_ENABLED']
# END - run_warp


# ============================================================================
def get_hdf_global_metadata(hdf_file):
    '''
    Description:
        Extract the metadata information from the HDF formatted file

    Note: Works with Ledaps and Modis generated HDF files
    '''

    cmd = ' '.join(['gdalinfo', hdf_file])
    output = utilities.execute_cmd(cmd)

    sb = StringIO()
    has_metadata = False
    for line in output.split('\n'):
        if str(line).strip().lower().startswith('metadata'):
            has_metadata = True
        if str(line).strip().lower().startswith('subdatasets'):
            break
        if str(line).strip().lower().startswith('corner'):
            break
        if has_metadata:
            sb.write(line.strip())
            sb.write('\n')

    sb.flush()
    metadata = sb.getvalue()
    sb.close()

    return metadata
# END - get_hdf_global_metadata


# ============================================================================
def hdf_has_subdatasets(hdf_file):
    '''
    Description:
        Determine if the HDF file has subdatasets
    '''

    cmd = ' '.join(['gdalinfo', hdf_file])
    output = utilities.execute_cmd(cmd)

    for line in output.split('\n'):
        if str(line).strip().lower().startswith('subdatasets'):
            return True

    return False
# END - hdf_has_subdatasets


# ============================================================================
def convert_hdf_to_gtiff(hdf_file):
    '''
    Description:
        Convert HDF formatted data to GeoTIFF
    '''

    logger = EspaLogging.get_logger('espa.processing')

    hdf_name = hdf_file.split('.hdf')[0]
    output_format = 'gtiff'

    logger.info("Retrieving global HDF metadata")
    metadata = get_hdf_global_metadata(hdf_file)
    if metadata is not None and len(metadata) > 0:
        metadata_filename = '%s-global_metadata.txt' % hdf_name

        logger.info("Writing global metadata to %s" % metadata_filename)
        with open(metadata_filename, 'w+') as metadata_fd:
            metadata_fd.write(str(metadata))

    # Extract the subdatasets into individual GeoTIFF files
    if hdf_has_subdatasets(hdf_file):
        for (sds_desc, sds_name) in parse_hdf_subdatasets(hdf_file):
            # Split the name into parts to extract the subdata name
            sds_parts = sds_name.split(':')
            subdata_name = sds_parts[len(sds_parts) - 1]
            # Quote the sds name due to possible spaces
            # Must be single because have double quotes in sds name
            quoted_sds_name = "'%s'" % sds_name
            no_data_value = get_no_data_value(quoted_sds_name)

            # Split the description into part to extract the string
            # which allows for determining the correct gdal data
            # data type, allowing specifying the correct no-data
            # value
            sds_parts = sds_desc.split('(')
            sds_parts = sds_parts[len(sds_parts) - 1].split(')')
            hdf_type = sds_parts[0]

            logger.info("Processing Subdataset %s" % quoted_sds_name)

            # Remove spaces from the subdataset name for the
            # final output name
            subdata_name = subdata_name.replace(' ', '_')
            output_filename = '%s-%s.tif' % (hdf_name, subdata_name)

            run_warp(quoted_sds_name, output_filename, output_format,
                     None, None, None, None,
                     None, None, 'near', no_data_value)

    # We only have the one dataset in the HDF file
    else:
        output_filename = '%s.tif' % hdf_name

        no_data_value = get_no_data_value(hdf_file)
        run_warp(hdf_file, output_filename, output_format,
                 None, None, None, None,
                 None, None, 'near', no_data_value)

    # Remove the HDF file, it is not needed anymore
    if os.path.exists(hdf_file):
        os.unlink(hdf_file)

    # Remove the associated hdr file
    hdr_filename = '%s.hdr' % hdf_file
    if os.path.exists(hdr_filename):
        os.unlink(hdr_filename)
# END - convert_hdf_to_gtiff


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

    logger = EspaLogging.get_logger('espa.processing')

    try:
        # Default the datum to WGS84
        datum = WGS84
        if parms['datum'] is not None:
            datum = parms['datum']

        bands = xml.get_bands()
        for band in bands.band:
            img_filename = band.get_file_name()
            hdr_filename = img_filename.replace('.img', '.hdr')
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
                gm.projection_information.set_datum(WGS84)  # WGS84 only
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
                gm.projection_information.set_datum(WGS84)  # WGS84 only
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
                sin_projection.set_sphere_radius(SINUSOIDAL_SPHERE_RADIUS)
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
            gm.projection_information.set_datum(WGS84)  # WGS84 only
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
def warp_to_geographic_with_subset(parms, i_filename=None, o_filename=None):
    '''
    Description:
      Warp to the geographic projection with subsetting.

    Notes:
      We use the original pixel size of the data for this.
    '''

    geo_projection_string = build_geographic_proj4_string()

    pixel_size = parms['data_pixel_size']
    if parms['data_pixel_size_units'] == 'meters':
        # Convert to degrees trying the defines ones first
        if pixel_size == 15.0:
            pixel_size = DEG_FOR_15_METERS
        elif pixel_size == 30.0:
            pixel_size = DEG_FOR_30_METERS
        else:
            pixel_size = (DEG_FOR_30_METERS / 30.0) * pixel_size

    run_warp(i_filename, o_filename, output_format='envi',
             min_x=parms['minx'],
             min_y=parms['miny'],
             max_x=parms['maxx'],
             max_y=parms['maxy'],
             pixel_size=pixel_size,
             projection=geo_projection_string,
             resample_method='near',
             no_data_value=parms['target_no_data_value'])
# END - warp_to_geographic_with_subset


# ============================================================================
def warp_to_target_without_subset(parms, i_filename=None, o_filename=None):

    run_warp(i_filename, o_filename, output_format='envi',
             min_x=None,
             min_y=None,
             max_x=None,
             max_y=None,
             pixel_size=parms['target_pixel_size'],
             projection=parms['target_proj4_projection'],
             resample_method=parms['resample_method'],
             no_data_value=parms['target_no_data_value'])
# END - warp_to_target_without_subset


# ============================================================================
def warp_to_target_with_subset(parms, i_filename=None, o_filename=None):

    run_warp(i_filename, o_filename, output_format='envi',
             min_x=parms['minx'],
             min_y=parms['miny'],
             max_x=parms['maxx'],
             max_y=parms['maxy'],
             pixel_size=parms['target_pixel_size'],
             projection=parms['target_proj4_projection'],
             resample_method=parms['resample_method'],
             no_data_value=parms['target_no_data_value'])
# END - warp_to_target_with_subset


# ============================================================================
def warp_image(parms, no_data_value=None,
               i_filename=None, o_filename=None):
    '''
    Description:
      Determine if the image needs to be warped to geographic first and then
      warp it appropriately
    '''

    # Might need to warp to geographic first
    target_projection = parms['target_projection']
    if (parms['image_extents_units'] == 'dd'
            and (target_projection is None or target_projection != 'lonlat')):

        # We need an in-between filename
        g_filename = 'geographic_warped.img'

        if target_projection is None:
            parms['target_proj4_projection'] = parms['data_proj4_projection']

        warp_to_geographic_with_subset(parms, i_filename, g_filename)

        warp_to_target_without_subset(parms, g_filename, o_filename)

        # Remove the .img
        os.unlink(g_filename)
        # Remove the .hdr
        hdr_filename = g_filename.replace('img', 'hdr')
        os.unlink(hdr_filename)
    else:
        warp_to_target_with_subset(parms, i_filename, o_filename)
# END - warp_image


# ============================================================================
def warp_espa_data(parms, scene, xml_filename=None):
    '''
    Description:
      Warp each espa science product to the parameters specified in the parms
    '''

    logger = EspaLogging.get_logger('espa.processing')

    # Validate the parameters
    validate_parameters(parms, scene)
    logger.debug(parms)

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

        # Get the proj4 projection string
        if parms['projection'] is not None:
            # Use the provided proj.4 projection string for the projection
            target_proj4_projection = parms['projection']
        elif parms['reproject']:
            # Verify and create proj.4 projection string
            target_proj4_projection = convert_target_projection_to_proj4(parms)
        else:
            # Default to the original
            target_proj4_projection = get_original_projection()

        parms['target_proj4_projection'] = target_proj4_projection

        # These will be poulated with the last bands information
        map_info_str = None

        # Process through the bands in the XML file
        for band in bands.band:
            img_filename = band.get_file_name()
            hdr_filename = img_filename.replace('.img', '.hdr')
            logger.info("Processing %s" % img_filename)

            # Figure out the pixel size to use
            pixel_size = parms['pixel_size']

            # EXECUTIVE DECISION(Calli) - ESPA Issue 185
            #    - If the band is Landsat 7 Band 8 do not resize the pixels.
            if satellite == 'LANDSAT_7' and band.get_name() == 'band8':
                if parms['target_projection'] == 'lonlat':
                    pixel_size = DEG_FOR_15_METERS
                else:
                    pixel_size = float(band.pixel_size.x)

            parms['data_pixel_size'] = float(band.pixel_size.x)
            parms['data_pixel_size_units'] = band.pixel_size.units
            parms['target_pixel_size'] = pixel_size

            # Open the image to read the no data value out since the internal
            # ENVI driver for GDAL does not output it, even if it is known
            ds = gdal.Open(img_filename)
            if ds is None:
                raise RuntimeError("GDAL failed to open (%s)" % img_filename)

            ds_band = None
            ds_srs = None
            try:
                ds_band = ds.GetRasterBand(1)
                ds_srs = osr.SpatialReference()
                ds_srs.ImportFromWkt(ds.GetProjection())
            except Exception, e:
                raise ee.ESPAException(ee.ErrorCodes.warping,
                                       str(e)), None, sys.exc_info()[2]

            # TODO - We don't process any floating point data types.... Yet
            # Save the no data value since gdalwarp does not write it out when
            # using the ENVI format
            no_data_value = ds_band.GetNoDataValue()
            if no_data_value is None:
                raise RuntimeError("no_data_value = None")
            else:
                # Convert to an integer then string
                no_data_value = str(int(no_data_value))

            parms['target_no_data_value'] = no_data_value

            # Save the source data's proj4 information
            parms['data_proj4_projection'] = ds_srs.ExportToProj4()

            del (ds_band)
            del (ds_srs)
            del (ds)

            tmp_img_filename = 'tmp-%s' % img_filename
            tmp_hdr_filename = 'tmp-%s' % hdr_filename

            warp_image(parms, no_data_value, img_filename, tmp_img_filename)

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
                        dummy = 'Nothing'
                    else:
                        sb.write(line)

                    if line.startswith('description'):
                        # This may be on multiple lines so read lines until
                        # found
                        if not line.strip().endswith('}'):
                            while 1:
                                next_line = tmp_fd.readline()
                                if (not next_line
                                        or next_line.strip().endswith('}')):
                                    break
                        sb.write('description = {ESPA-generated file}\n')
                    elif line.startswith('data type'):
                        sb.write('data ignore value = %s\n' % no_data_value)
                    elif line.startswith('map info'):
                        map_info_str = line
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

    logger = EspaLogging.get_logger('espa.processing')

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
    EspaLogging.configure('espa.processing', order='test',
                          product='product', debug=args.debug)
    logger = EspaLogging.get_logger('espa.processing')

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
