# im3vis
General plotting functions for IM3 modeling needs

## Get started with `im3vis`

Install `im3vis` from GitHub using:
```bash
python -m pip install git+https://github.com/IMMM-SFA/im3vis.git
```

Import the `im3vis` package into your local Python environment
```python
import im3vis
```

## Current functionality


### GCAM

#### Plot GCAM total land allocation by region for all land classes 
```python
reg_ax = im3vis.gcam_demeter_region(gcam_df, target_year='2015')
```

#### Plot GCAM total land allocation by region for combined Corn
```python
reg_ax = im3vis.gcam_demeter_region(gcam_df, 
                                    target_year='2015', 
                                    landclass_list=['corn_irr', 'corn_rfd'])
```

#### Plot map of GCAM `corn` allocation for year 2015 for the CONUS
```python
agg_df = im3vis.plot_gcam_basin(gcam_df,
                                target_year='2015',
                                landclass_list=['corn_irr', 'corn_rfd'],
                                setting='crop_yield',
                                scope='conus')
```

#### Plot map of GCAM `corn` allocation for year 2015 for global basins
```python
agg_df = im3vis.plot_gcam_basin(gcam_df,
                                target_year='2015',
                                landclass_list=['corn_irr', 'corn_rfd'],
                                setting='crop_yield',
                                scope='global')
```

### Demeter

#### Build a GeoPandas data frame of Demeter's output land allocation data with geometry
```python
demeter_gdf = im3vis.build_geodataframe(demeter_2015)
```

#### Plot gridded Demeter map of `corn` output for year 2015 for the CONUS at 0.5 degree resolution
```python
r = im3vis.plot_demeter_raster(demeter_gdf=demeter_gdf, 
                               landclass_list=['crop2_irr', 'crop2_rfd'],
                               target_year='2015', 
                               scope='conus',
                               resolution='0.5')
```

#### Plot gridded Demeter map of `forest` output for year 2015 for the globe at 0.5 degree resolution
```python
r = im3vis.plot_demeter_raster(demeter_gdf=demeter_gdf, 
                               landclass_list=['unmanagedforest', 'forest'],
                               target_year='2015', 
                               scope='global',
                               resolution='0.5')
```
