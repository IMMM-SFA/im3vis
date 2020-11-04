import tempfile
import pkg_resources

import pandas as pd
import seaborn as sns; sns.set()
import geopandas as gpd
import rasterio

from shapely.geometry import Point
from rasterio import features
from rasterio.plot import show
from matplotlib import pyplot as plt


def gcam_demeter_region(df, target_year, figure_size=(12, 8), metric_id_col='metric_id', region_col='region',
                        landclass_col='landclass', landclass_list=None, font_scale=1.5):
    """Create a bar plot of GCAM land allocation per each geopolitical region.  Optionally, do
    so for a specific land class.

    :param df:                      Input data frame generated from the GCAM projected input for Demeter.
    :type df:                       data frame

    :param target_year:             Target year column name to examine as a four digit year (e.g., 2000) cast as a string
    :type target_year:              str

    :param figure_size:             x and y integer size for the resulting figure
    :type figure_size:              tuple

    :param metric_id_col:           name of the metric id column in the input data frame
    :type metric_id_col:            str

    :param region_col:              name of the region name column in the input data frame
    :type metric_id_col:            str

    :param landclass_col:           name of the landclass name column in the input data frame
    :type metric_id_col:            str

    :param landclass_list:          a list of land class names to aggregate by region
    :type landclass_list:           list

    :param font_scale:              font scaling factor for seaborn set class
    :type font_scale:               float

    :return:                        axis object

    """
    sns.set(font_scale=font_scale)

    # create a copy of the input data frame
    gcam_reg_df = df.copy()

    # set up plot with custom size
    fig, ax = plt.subplots(figsize=figure_size)

    # extract specific land classes
    if landclass_list is None:
        title_adder = ''
    else:
        gcam_reg_df = gcam_reg_df.loc[gcam_reg_df[landclass_col].isin(landclass_list)]
        title_adder = f"for land classes {','.join(landclass_list)}"

    # drop unneeded fields
    gcam_reg_df.drop(metric_id_col, inplace=True, axis=1)

    # sum by region
    gcam_reg_df = gcam_reg_df.groupby(region_col).sum()

    # sort the data frame by allocation
    gcam_reg_df.sort_values(by=[target_year], inplace=True)

    # generate plot
    g = sns.barplot(x=gcam_reg_df.index, y=target_year, data=gcam_reg_df[[target_year]])
    g.set(xlabel='GCAM Geopolitical Region',
          ylabel=f"{target_year} land allocation (thous km)",
          title=f"GCAM land allocation by region{title_adder}")
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    return g


def plot_conus_raster(boundary_gdf, demeter_gdf, landclass, target_year, font_scale=1.5):
    """Generate a raster plot from demeter outputs for the CONUS for a specified land class."""

    sns.set(font_scale=font_scale)

    # generate a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.tif')
    rast = temp_file.name

    # create a generator of geom, value pairs to use in rasterizing
    shapes = ((geom, value) for geom, value in zip(demeter_gdf.geometry, demeter_gdf[landclass]))

    metadata = get_conus_metadata()

    # burn point values in to raster
    with rasterio.open(rast, 'w+', **metadata) as out:
        out_arr = out.read(1)

        burned = features.rasterize(shapes=shapes, fill=metadata['nodata'], out=out_arr, transform=out.transform)
        out.write_band(1, burned)

        # open and visualize
    with rasterio.open(rast) as src:
        fig, ax = plt.subplots(1, figsize=(10, 4))

        boundary_gdf.geometry.boundary.plot(ax=ax, color='grey', lw=0.4)

        show(src,
             cmap='YlGn',
             ax=ax,
             title=f"Demeter land allocation for {landclass} for {target_year}")

        return src


def build_geodataframe(demeter_file, longitude_col='longitude', latitude_col='latitude', crs='epsg:4326'):
    """Build a geodataframe from a pandas data frame containing coordinates"""

    # read in as pandas data frame
    df = pd.read_csv(demeter_file)

    # create geometry column from coordinate fields
    geometry = [Point(xy) for xy in zip(df[longitude_col], df[latitude_col])]

    # coordinate reference system for WGS84
    return gpd.GeoDataFrame(df, crs=crs, geometry=geometry)


def get_conus_metadata():
    """Get CONUS metadata from template raster."""

    template_raster = pkg_resources.resource_filename('im3vis', 'data/demeter_conus_template.tif')

    r = rasterio.open(template_raster)
    meta = r.meta.copy()
    meta.update(compress='lzw')

    return meta


def plot_gcam_conus_basin(gcam_df, target_year, landclass):
    """Generate a plot of GCAM land allocation by basin for the CONUS."""

    gxf = gpd.read_file(pkg_resources.resource_filename('im3vis', 'data/conus_basins.shp'))

    lc_mapping = gcam_to_demeter_lc_map()

    # only account for forest mapping for demonstration
    gcam_df['demeter_lc'] = gcam_df['landclass'].map(lc_mapping)

    # only keep forest classes in the USA
    gcam_df = gcam_df.loc[(gcam_df['demeter_lc'] == landclass) & (gcam_df['region'] == 'USA')].copy()

    # only keep what we need
    gcam_us = gcam_df[['metric_id', 'demeter_lc', target_year]].copy()

    # group by metric id
    grp_us = gcam_us.groupby('metric_id').sum()
    grp_us.reset_index(inplace=True)

    # rename metric id field
    grp_us.rename(columns={'metric_id': 'basin_id'}, inplace=True)

    # merge with spatial boundaries
    mdf = gxf.merge(grp_us, on='basin_id')

    fig, ax = plt.subplots(1, 1)
    mdf.plot(column=target_year,
             ax=ax,
             legend=True,
             figsize=(15, 10),
             edgecolor='grey',
             legend_kwds={'label': f"GCAM land allocation for {landclass} in {target_year} (thous km)",
                          'orientation': "horizontal"},
             cmap='viridis')

    return mdf


def plot_gcam_reclassified(gcam_df, landclass, start_yr=2015, through_yr=2100, interval=5):

    """Plot GCAM landclass allocation over a time period reclassified to Demeter land cover types."""

    yrs = [str(i) for i in range(start_yr, through_yr + interval, interval)]

    lc_mapping = gcam_to_demeter_lc_map()

    # US corn allocation through the end of the century
    us_df = gcam_df.loc[gcam_df['region'] == 'USA'].copy()

    # bin gcam land classes into demeter land classes
    us_df['demeter_lc'] = us_df['landclass'].map(lc_mapping)

    # group by demeter crop assignment
    us_grp = us_df.groupby('demeter_lc').sum()

    # get only target lc
    us_grp = us_grp.loc[us_grp.index == landclass].copy()

    # plot GCAM
    g = us_grp[yrs].T.plot(legend=False)

    g.set(xlabel='Year',
          ylabel=f"Land allocation (thous km)",
          title=f"GCAM land allocation for {landclass} from {yrs[0]} through {yrs[-1]}")
    g.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    return us_grp


def gcam_to_demeter_lc_map():
    """Mapping from GCAM land classes to Demeter land classes."""

    return {'biomass': 'crops',
                  'Corn': 'crops',
                  'FiberCrop': 'crops',
                  'FodderGrass': 'crops',
                  'FodderHerb': 'crops',
                  'Forest': 'forest',
                  'ProtectedUnmanagedForest': 'forest',
                  'Grassland': 'grass',
                  'ProtectedGrassland': 'grass',
                  'MiscCrop': 'crops',
                  'OilCrop': 'crops',
                  'OtherArableLand': 'crops',
                  'OtherGrain': 'crops',
                  'PalmFruit': 'crops',
                  'Pasture': 'grass',
                  'ProtectedUnmanagedPasture': 'grass',
                  'Rice': 'crops',
                  'Root_Tuber': 'crops',
                  'Shrubland': 'shrub',
                  'ProtectedShrubland': 'shrub',
                  'Tundra': 'grass',
                  'RockIceDesert': 'sparse',
                  'SugarCrop': 'crops',
                  'UnmanagedForest': 'forest',
                  'UnmanagedPasture': 'grass',
                  'UrbanLand': 'urban',
                  'Wheat': 'crops'}
