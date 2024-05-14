<div style="color: #2F5496; text-decoration: underline; font-size: 20px; font-weight: bold;">
    Data
</div>

**Below are links to the raw datafiles used in this analysis. Each requires downloading and saving locally. If saved alongside this Notebook, this must be in a folder called 'Raw data files' to use the code in this Notebook without edits.**

Office for National Statistics (ONS). 2021. *Census 2021 geographies*. [Online]. [Accessed 09 January 2024]. Available from: https://www.ons.gov.uk/methodology/geography/ukgeographies/censusgeographies/census2021geographies

Office for National Statistics (ONS). 2023a. Output Areas (2021) Boundaries EW BFC. *Open Geography Portal*. [Online]. [Accessed 11 December 2023]. Available from: https://geoportal.statistics.gov.uk/datasets/ons::output-areas-2021-boundaries-ew-bfc/about

Office for National Statistics (ONS). 2023b. Output Areas (December 2021) PWC (V3). *Open Geography Portal*. [Online]. [Accessed 11 December 2023]. Available from: https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-pwc-v3/about

Office for National Statistics (ONS). 2023c. Output Area to Lower layer Super Output Area to Middle layer Super Output Area to Local Authority District (December 2021) Lookup in England and Wales V2. *Open Geography Portal*. [Online]. [Accessed 14 December 2023]. Available from: https://geoportal.statistics.gov.uk/datasets/output-area-to-lower-layer-super-output-area-to-middle-layer-super-output-area-to-local-authority-district-december-2021-lookup-in-england-and-wales-v2-1/about

Office for National Statistics (ONS). 2023d. Household deprivation in the education dimension. *Office for National Statistics*. [Online]. [Accessed 14 December 2023]. Available from: https://www.ons.gov.uk/filters/d7bede85-c97b-4b85-84a8-87b5164d3add/dimensions

Office for National Statistics (ONS). 2023e. Household deprivation in the employment dimension. *Office for National Statistics*. [Online]. [Accessed 14 December 2023]. Available from: https://www.ons.gov.uk/filters/103a7748-96d9-4e79-97e2-e1cc42ef7024/dimensions

Office for National Statistics (ONS). 2023f. Household deprivation in the health dimension. *Office for National Statistics*. [Online]. [Accessed 14 December 2023]. Available from: https://www.ons.gov.uk/filters/15149637-c729-4aa5-b817-d687cad093d8/dimensions

Office for National Statistics (ONS). 2023g. Household deprivation in the housing dimension. *Office for National Statistics*. [Online]. [Accessed 14 December 2023]. Available from: https://www.ons.gov.uk/filters/cf7beddc-198f-411a-9b66-9c690d18e3bf/dimensions

Ordnance Survey (OS). 2023. OS Open Greenspace. *Ordnance Survey*. [Online]. [Accessed 6 December 2023]. Available from: https://osdatahub.os.uk/downloads/open/OpenGreenspace

<div style="color: #2F5496; text-decoration: underline; font-size: 18px; font-weight: bold;">
    Import libraries
</div>


```python
# Set number of threads to recommended 7 to avoid errors in code during visualisation phase
import os
os.environ["OMP_NUM_THREADS"] = '7'
```


```python
# Dataframe libraries
import pandas as pd
import geopandas as gpd
```

<div style="color: #2F5496; text-decoration: underline; font-size: 18px; font-weight: bold;">
    Import each datafile
</div>

<div style="color: #2F5496; font-size: 14px; font-weight: bold;">
    Output Areas (OAs)
</div>

OAs are the lowest census geography level, containing 40 to 250 households (ONS, 2021), enabling more accurate nearest greenspace measurements, and capturing localised differences in deprivation. Polygons and population weighted centroids (PWCs) are available from the ONS (2023a, 2023b). PWCs are used to measure distance to greenspace as it captures the nearest greenspace for most households. Polygons are aggregated to Bradford LAD level using the ONS (2023c) lookup. This is necessary to reduce the OA PWCs and greenspace to those within Bradford LAD.


```python
# OA polygons (BFC: Full resolution - clipped to the coastline (Mean High Water mark)) (ONS, 2023a)
OA_polygons = gpd.read_file('Raw data files/Output_Areas_2021_EW_BFC_V8/OA_2021_EW_BFC_V8.shp')

# OA Population-weighted centroids (ONS, 2023b)
OA_PWC = gpd.read_file('Raw data files/Output_Areas_2021_PWC_V3/PopCentroids_EW_2021_V3.shp')
```


```python
# OA to LSOA to MSOA to LAD lookup (ONS, 2023c)
OA_lookup = pd.read_csv('Raw data files/Output_Area_Lookup_in_England_and_Wales_v3.csv')

### Error can be ignored - it is because of Welsh spellings in a certain column.
### This is not relevant to, nor will it affect, this study as these columns/rows will not be used.
```

    C:\Users\jessa\AppData\Local\Temp\ipykernel_33132\3658327603.py:2: DtypeWarning: Columns (3,6,9) have mixed types. Specify dtype option on import or set low_memory=False.
      OA_lookup = pd.read_csv('Raw data files/Output_Area_Lookup_in_England_and_Wales_v3.csv')
    

<div style="color: #2F5496; font-size: 14px; font-weight: bold;">
    Greenspace
</div>

Greenspace data (OS, 2023) include site (polygon) and access (point) data. However, not all sites have access data, hence only sites are used to ensure all eligible greenspace is captured. Distance from the PWC to nearest greenspace will be calculated using these polygons. Data includes parks, gardens, sports facilities, and religious grounds, but does not include forests, woodland, moorland or canal and river paths (OS, 2023).


```python
# Greenspace site polygons (OS, 2023)
GB_greenspace_site = gpd.read_file('Raw data files/OS Open Greenspace (ESRI Shape File) GB/data/GB_GreenspaceSite.shp')
```

<div style="color: #2F5496; font-size: 14px; font-weight: bold;">
    Independent variables
</div>

Each deprivation dimension (education, employment, health and disability, and housing) can be obtained via custom ONS queries (ONS, 2023d; ONS, 2023e; ONS, 2023f; ONS, 2023g). These queries include only Bradford LAD OAs. Each file contains the number of households that are deprived or not deprived, from which percentage of households deprived in each dimension is calculated (Section 5.4.) creating the independent variables for regression modelling.


```python
# Data for Bradford OAs only, from 2021 Census

# Household deprivation in the education dimension (ONS, 2023d)
HHD_dep_education_raw = pd.read_csv('Raw data files/2021 Census Data/Household deprived in the education dimension.csv')

# Household deprivation in the employment dimension (ONS, 2023e)
HHD_dep_employment_raw = pd.read_csv('Raw data files/2021 Census Data/Household deprived in the employment dimension.csv')

# Household deprivation in the health and disability dimension (ONS, 2023f)
HHD_dep_health_raw = pd.read_csv('Raw data files/2021 Census Data/Household deprived in the health and disability dimension.csv')

# Household deprivation in the housing dimension (ONS, 2023g)
HHD_dep_housing_raw = pd.read_csv('Raw data files/2021 Census Data/Household deprived in the housing dimension.csv')
```

<div style="color: #2F5496; text-decoration: underline; font-size: 20px; font-weight: bold;">
    Data wrangling
</div>

<div style="color: #2F5496; text-decoration: underline; font-size: 18px; font-weight: bold;">
    OAs
</div>

Only Bradford LAD OAs are needed. The OA lookup and geometry dataframes can be merged, retaining only Bradford as identified by the LAD22NM column. There are 1,575 OAs within Bradford LAD.


```python
# Remove unnecessary columns from OA lookup, keeping only required columns in new dataframe
OA_lookup_trim = OA_lookup[['OA21CD','LAD22CD','LAD22NM']]
```


```python
# Filter OA lookup to just Bradford
Bradford_OA_lookup = OA_lookup_trim[OA_lookup_trim['LAD22NM'] == 'Bradford']
```


```python
# Check how many OAs are left - this number is an important reference point for further data wrangling tasks
len(Bradford_OA_lookup)
```




    1575




```python
# Merge Bradford_OA_lookup with OA polygons to reduce the geodataframe containing the polygons to only Bradford OAs
Bradford_OA_polygons = OA_polygons.merge(Bradford_OA_lookup, how='right', on='OA21CD')
```


```python
# Check number of OAs is same as in the reduced Bradford OA lookup dataframe
print(len(Bradford_OA_polygons))
print(len(Bradford_OA_polygons) == len(Bradford_OA_lookup))
```

    1575
    True
    


```python
# Merge Bradford_OA_lookup with OA PWCs to reduce the geodataframe containing the PWCs to only Bradford OAs
Bradford_OA_PWC = OA_PWC.merge(Bradford_OA_lookup, how='right', on='OA21CD')
```


```python
# Check number of OAs is same as in the reduced Bradford OA lookup dataframe
print(len(Bradford_OA_PWC))
print(len(Bradford_OA_PWC) == len(Bradford_OA_lookup))
```

    1575
    True
    


```python
# Drop unrequired columns
Bradford_OA_polygons_tidy = Bradford_OA_polygons.drop([
    'LSOA21CD','LSOA21NM','LSOA21NMW','BNG_E','BNG_N','LAT','LONG','GlobalID','LAD22CD','LAD22NM'], axis=1)
Bradford_OA_PWC_tidy = Bradford_OA_PWC.drop(['GlobalID','LAD22CD','LAD22NM'], axis=1)

# Rename geometry columns and set geometry
Bradford_OA_polygons = Bradford_OA_polygons_tidy.set_geometry('geometry').rename_geometry('Polygon')
Bradford_OA_PWC = Bradford_OA_PWC_tidy.set_geometry('geometry').rename_geometry('PWC')
```

<div style="color: #2F5496; text-decoration: underline; font-size: 18px; font-weight: bold;">
    Greenspace
</div>

Only greenspace within Bradford LAD is required. A Bradford LAD polygon is created to spatially match the greenspace, retaining only sites within Bradford LAD.

<div style="color: #2F5496; font-size: 14px; font-weight: bold;">
    Clean greenspace dataframe
</div>

Some types of greenspaces are arguably not for recreation (e.g. cemetery) and some are not always free to the public (e.g. golf course). To ensure this analysis includes only freely accessible greenspace for recreational use, types are reduced to 'play space', 'playing field' and 'public park or garden'.


```python
# Remove unnecessary columns, keeping only required columns in new dataframe
GB_greenspace_site_trim = GB_greenspace_site[['id','function','geometry']]

# Rename remaining columns for clarity
GB_greenspace_site_trim = GB_greenspace_site_trim.rename(columns={"id": "Greenspace_ID", "function": "Type"})
GB_greenspace_site_trim = GB_greenspace_site_trim.set_geometry('geometry').rename_geometry('Polygon')
```


```python
# Check what types are included in the data
GB_greenspace_site_trim.Type.value_counts()
```




    Type
    Play Space                                42972
    Religious Grounds                         22229
    Playing Field                             21377
    Other Sports Facility                     15073
    Allotments Or Community Growing Spaces    13002
    Public Park Or Garden                     11982
    Cemetery                                   7559
    Tennis Court                               6632
    Bowling Green                              6589
    Golf Course                                3000
    Name: count, dtype: int64




```python
# Keep only 'Play Space', 'Playing Field', and 'Public Park Or Garden'
GB_greenspace_site_filtered = GB_greenspace_site_trim.loc[GB_greenspace_site_trim['Type'].
                                                          isin(['Play Space','Playing Field','Public Park Or Garden'])]
```

<div style="color: #2F5496; font-size: 14px; font-weight: bold;">
    Greenspace within Bradford LAD
</div>

Aggregating OA polygons to a single polygon and spatially matching to the greenspace dataframe isolates greenspace within Bradford LAD. This will be used to calculate the dependent variable for the regression analysis.


```python
# Create a copy of the Bradford_OA_polygons dataframe which will become the Bradford polygon dataframe
Bradford = Bradford_OA_polygons
# Add a 'Bradford' dummy column to aggregate on
Bradford['City'] = 'Bradford'
```


```python
# Create a single polygon aggregated to the city level using 'dissolve'
Bradford_polygon = Bradford.dissolve(by='City')
```


```python
# Drop OA21CD column as this is no longer relevant
Bradford_polygon = Bradford_polygon.drop(['OA21CD'], axis=1)
```


```python
# Spatial join for any greenspace that intersects the Bradford polygon
Bradford_greenspace_site = gpd.sjoin(GB_greenspace_site_filtered, Bradford_polygon, how='inner', predicate='intersects')
```


```python
# Drop index_right column as this is no longer relevant
Bradford_greenspace_site = Bradford_greenspace_site.drop(['index_right'], axis=1)
```


```python
# Check length of dataframe to determine how many greenspaces are included
len(Bradford_greenspace_site)
```




    436




```python
# Check that the unique number of Greenspace_IDs is in fact the length of the dataframe
print(Bradford_greenspace_site['Greenspace_ID'].nunique())
print(len(Bradford_greenspace_site) == Bradford_greenspace_site['Greenspace_ID'].nunique())
```

    436
    True
    

<div style="color: #2F5496; text-decoration: underline; font-size: 18px; font-weight: bold;">
    Dependent variable
</div>

The dependent variable is the distance between the OA PWC and nearest greenspace polygon calculated using the Euclidian distance and given in metres.


```python
# Find the nearest greenspace to each OA population-weighted centroid
nearest_greenspace = gpd.sjoin_nearest(
    Bradford_OA_PWC, Bradford_greenspace_site, how='left', distance_col='Distance')

# Drop the unrequired columns
nearest_greenspace = nearest_greenspace.drop(['PWC','index_right'], axis=1)
```


```python
# Check length of dataframe is equal to the length of the OA dataframe - i.e. one greenspace per OA has been identified
print(len(nearest_greenspace))
print(len(nearest_greenspace) == len(Bradford_OA_lookup))
```

    1599
    False
    

The information above shows there are some duplicated rows. This is confirmed by checking the number of unique OA21CDs which should be 1,575.


```python
# Check number of OA21CDs
print(nearest_greenspace['OA21CD'].nunique())
print(nearest_greenspace['OA21CD'].nunique() == len(Bradford_OA_lookup))
```

    1575
    True
    

Duplicates are caused by the same greenspace being allocated different 'Types' with unique IDs. These duplicates are removed, keeping the first record irrespective of 'Type' as no analysis is to be conducted on this.


```python
# Isolate duplicated rows and check cause
# Confirmed as duplicated sites with different Greenspace_IDs due to different greenspace Type being recorded
duplicated_rows = nearest_greenspace[nearest_greenspace.duplicated(subset='OA21CD', keep=False)]
duplicated_rows.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OA21CD</th>
      <th>Greenspace_ID</th>
      <th>Type</th>
      <th>Distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>E00053364</td>
      <td>0295ED18-F337-5C37-E063-AAEFA00A445E</td>
      <td>Play Space</td>
      <td>514.057891</td>
    </tr>
    <tr>
      <th>11</th>
      <td>E00053364</td>
      <td>0295ED18-F2F3-5C37-E063-AAEFA00A445E</td>
      <td>Playing Field</td>
      <td>514.057891</td>
    </tr>
    <tr>
      <th>42</th>
      <td>E00053392</td>
      <td>0295ED18-F337-5C37-E063-AAEFA00A445E</td>
      <td>Play Space</td>
      <td>364.166159</td>
    </tr>
    <tr>
      <th>42</th>
      <td>E00053392</td>
      <td>0295ED18-F2F3-5C37-E063-AAEFA00A445E</td>
      <td>Playing Field</td>
      <td>364.166159</td>
    </tr>
    <tr>
      <th>43</th>
      <td>E00053393</td>
      <td>0295ED18-F337-5C37-E063-AAEFA00A445E</td>
      <td>Play Space</td>
      <td>571.723915</td>
    </tr>
    <tr>
      <th>43</th>
      <td>E00053393</td>
      <td>0295ED18-F2F3-5C37-E063-AAEFA00A445E</td>
      <td>Playing Field</td>
      <td>571.723915</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop duplicated rows and keep first row for duplicated instances
unique_nearest_greenspace = nearest_greenspace.drop_duplicates(subset='OA21CD', keep='first')
```


```python
# Check length of dataframe is now as expected
print(len(unique_nearest_greenspace))
print(len(unique_nearest_greenspace) == len(Bradford_OA_lookup))
```

    1575
    True
    

The 'Bradford_greenspace_site' dataframe must also be updated to ensure a consistent view of greenspace.


```python
# Get unique list of Greenspace_IDs
unique_Greenspace_IDs = unique_nearest_greenspace[['Greenspace_ID']].drop_duplicates()
```


```python
# Join onto Bradford_greenspace_site to reduce this dataframe to just those kept in the unique_nearest_greenspace dataframe
Bradford_greenspace_site_unique = Bradford_greenspace_site.merge(unique_Greenspace_IDs, how='right', on='Greenspace_ID')
```

<div style="color: #2F5496; text-decoration: underline; font-size: 18px; font-weight: bold;">
    Independent variables
</div>

Deprivation data needs to be pivoted to create columns of the datapoints by OA and converting these into percentages. This is achieved using a function to repeat the process for each dimension. Resulting dataframes are merged into a final dataframe containing all four independent variables.


```python
def Independent_Variables_Setup(dataframe, dataframe_name):     # dataframe as object, dataframe_name as string
    
    # Create all required strings for naming dataframes throughout wrangling process
    core_name = dataframe_name.rstrip('_raw') # Remove "_raw" from end of dataframe to get core name
    dropped = core_name + '_dropped'          # Add "_dropped" to end of core name
    pivot = core_name + '_pivot'              # Add "_pivot" to end of core name
    percentage = core_name + '_PC'            # Add "_PC" to end of core name
    
    # Drop columns that are not required
    cols = [1,3]     # List of column indexes of the columns to drop
    dropped = dataframe.drop(dataframe.columns[cols], axis=1)     # Create new dataframe with columns dropped
    
    # Pivot the table on column[0], making the options from column[1] the new column headers and the data that from column[2]
    pivot = dropped.pivot_table(index=dropped.columns[0],
                                  columns=dropped.columns[1],
                                  values=dropped.columns[2]).reset_index()
    pivot.columns.name = None
    
    # Create a Total column from 3 new data columns[1,2,3]
    pivot['Total'] = pivot.iloc[:, 1:4].sum(axis=1)
    
    # Calculate % HHDs Deprived by dividing column[3] "Deprived" by the Total calculated above
    pivot[percentage] = (pivot.iloc[:,3]/pivot['Total'])
    
    # Create final dataframe keeping only the OA21CD and % HHDs deprived in the given dimension
    final_cols = [0,5]
    core_name = pivot.iloc[:, final_cols]
    
    return core_name
```


```python
# Household deprivation in the education dimension
HHD_dep_education = Independent_Variables_Setup(HHD_dep_education_raw, 'HHD_dep_education_raw')
```


```python
# Household deprivation in the employment dimension
HHD_dep_employment = Independent_Variables_Setup(HHD_dep_employment_raw, 'HHD_dep_employment_raw')
```


```python
# Household deprivation in the health and disability dimension
HHD_dep_health = Independent_Variables_Setup(HHD_dep_health_raw, 'HHD_dep_health_raw')
```


```python
# Household deprivation in the housing dimension
HHD_dep_housing = Independent_Variables_Setup(HHD_dep_housing_raw, 'HHD_dep_housing_raw')
```


```python
# Check all 4 dataframes are the same length and that length is as expected
len(HHD_dep_education) == len(HHD_dep_employment) == len(HHD_dep_health) == len(HHD_dep_housing) == len(Bradford_OA_lookup)
```




    True




```python
# Merge education and employment
Ind_vars = HHD_dep_education.merge(HHD_dep_employment, how='inner', on='Output Areas Code')
# Add health
Ind_vars = Ind_vars.merge(HHD_dep_health, how='inner', on='Output Areas Code')
# Add housing
Ind_vars = Ind_vars.merge(HHD_dep_housing, how='inner', on='Output Areas Code')
```


```python
# Final length check
print(len(Ind_vars))
print(len(Ind_vars) == len(Bradford_OA_lookup))
```

    1575
    True
    

<div style="color: #2F5496; text-decoration: underline; font-size: 18px; font-weight: bold;">
    Final dataframe for regression analysis
</div>

A dataframe containing the dependent and independent variables, OA code and polygons is required for the regression models and spatial analysis.


```python
# Merge the unique_nearest_greenspace and Ind_vars dataframes to get a final dataframe for subsequent analysis
final_df = unique_nearest_greenspace.merge(Ind_vars, how='inner', left_on='OA21CD', right_on='Output Areas Code')
final_df = final_df.merge(Bradford_OA_polygons, how='inner', on='OA21CD')

# Drop unrequired columns
final_df = final_df.drop(['Output Areas Code', 'Greenspace_ID', 'Type', 'City'], axis=1)

# Set geometry
final_df = final_df.set_geometry('Polygon')
```

<div style="color: #2F5496; text-decoration: underline; font-size: 18px; font-weight: bold;">
    Export dataframes
</div>

Visual checks of each dataframe and export as GeoJSON or csv as appropriate.

#### Bradford OA polygons dataframe


```python
Bradford_OA_polygons.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OA21CD</th>
      <th>Polygon</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E00053353</td>
      <td>POLYGON ((415817.093 440872.597, 415821.094 44...</td>
      <td>Bradford</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E00053354</td>
      <td>POLYGON ((415078.000 439967.001, 415058.323 43...</td>
      <td>Bradford</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E00053355</td>
      <td>POLYGON ((416252.367 439816.041, 416253.270 43...</td>
      <td>Bradford</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E00053356</td>
      <td>POLYGON ((416668.000 439392.028, 416667.653 43...</td>
      <td>Bradford</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E00053357</td>
      <td>POLYGON ((415143.909 439176.235, 415143.000 43...</td>
      <td>Bradford</td>
    </tr>
  </tbody>
</table>
</div>




```python
Bradford_OA_polygons.to_file('Wrangled dataframes/Bradford_OA_polygons.geojson', driver='GeoJSON')
```

#### Bradford OA population weighted centroids dataframe


```python
Bradford_OA_PWC.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OA21CD</th>
      <th>PWC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E00053353</td>
      <td>POINT (413638.052 439495.615)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E00053354</td>
      <td>POINT (414837.013 439813.246)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E00053355</td>
      <td>POINT (416162.559 439674.009)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E00053356</td>
      <td>POINT (416591.137 439417.227)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E00053357</td>
      <td>POINT (414671.681 439110.823)</td>
    </tr>
  </tbody>
</table>
</div>




```python
Bradford_OA_PWC.to_file('Wrangled dataframes/Bradford_OA_PWC.geojson', driver='GeoJSON')
```

#### Bradford greenspace polygons (unique list)


```python
Bradford_greenspace_site_unique.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Greenspace_ID</th>
      <th>Type</th>
      <th>Polygon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0295ED18-D538-5C37-E063-AAEFA00A445E</td>
      <td>Playing Field</td>
      <td>POLYGON Z ((414018.070 438415.690 0.000, 41399...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0295ED18-E5D4-5C37-E063-AAEFA00A445E</td>
      <td>Playing Field</td>
      <td>POLYGON Z ((415193.700 439129.550 0.000, 41519...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0295ED18-F2F3-5C37-E063-AAEFA00A445E</td>
      <td>Playing Field</td>
      <td>POLYGON Z ((415690.440 439919.060 0.000, 41568...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0295ECC9-0C25-5C37-E063-AAEFA00A445E</td>
      <td>Play Space</td>
      <td>POLYGON Z ((416739.570 439592.210 0.000, 41675...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0295ECC7-FBFC-5C37-E063-AAEFA00A445E</td>
      <td>Play Space</td>
      <td>POLYGON Z ((415991.960 438808.610 0.000, 41598...</td>
    </tr>
  </tbody>
</table>
</div>




```python
Bradford_greenspace_site_unique.to_file('Wrangled dataframes/Bradford_greenspace_site_unique.geojson', driver='GeoJSON')
```

#### Nearest greenspace to each OA (unique list)


```python
unique_nearest_greenspace.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OA21CD</th>
      <th>Greenspace_ID</th>
      <th>Type</th>
      <th>Distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E00053353</td>
      <td>0295ED18-D538-5C37-E063-AAEFA00A445E</td>
      <td>Playing Field</td>
      <td>1000.455806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E00053354</td>
      <td>0295ED18-E5D4-5C37-E063-AAEFA00A445E</td>
      <td>Playing Field</td>
      <td>771.145845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E00053355</td>
      <td>0295ED18-F2F3-5C37-E063-AAEFA00A445E</td>
      <td>Playing Field</td>
      <td>418.265674</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E00053356</td>
      <td>0295ECC9-0C25-5C37-E063-AAEFA00A445E</td>
      <td>Play Space</td>
      <td>194.292561</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E00053357</td>
      <td>0295ED18-E5D4-5C37-E063-AAEFA00A445E</td>
      <td>Playing Field</td>
      <td>432.352481</td>
    </tr>
  </tbody>
</table>
</div>




```python
unique_nearest_greenspace.to_csv('Wrangled dataframes/unique_nearest_greenspace.csv', index=False)
```

#### Independent variables dataframe


```python
Ind_vars.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Output Areas Code</th>
      <th>HHD_dep_education_PC</th>
      <th>HHD_dep_employment_PC</th>
      <th>HHD_dep_health_PC</th>
      <th>HHD_dep_housing_PC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E00053353</td>
      <td>0.282787</td>
      <td>0.114754</td>
      <td>0.336066</td>
      <td>0.077869</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E00053354</td>
      <td>0.172932</td>
      <td>0.067669</td>
      <td>0.308271</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E00053355</td>
      <td>0.112000</td>
      <td>0.048780</td>
      <td>0.219512</td>
      <td>0.016000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E00053356</td>
      <td>0.215278</td>
      <td>0.041667</td>
      <td>0.340278</td>
      <td>0.048611</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E00053357</td>
      <td>0.141844</td>
      <td>0.034965</td>
      <td>0.212766</td>
      <td>0.007092</td>
    </tr>
  </tbody>
</table>
</div>




```python
Ind_vars.to_csv('Wrangled dataframes/Ind_vars.csv', index=False)
```

#### Bradford polygon


```python
Bradford_polygon
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Polygon</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bradford</th>
      <td>POLYGON ((404750.688 432366.281, 404708.240 43...</td>
    </tr>
  </tbody>
</table>
</div>




```python
Bradford_polygon.to_file('Wrangled dataframes/Bradford_polygon.geojson', driver='GeoJSON')
```

#### Final dataframe


```python
final_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OA21CD</th>
      <th>Distance</th>
      <th>HHD_dep_education_PC</th>
      <th>HHD_dep_employment_PC</th>
      <th>HHD_dep_health_PC</th>
      <th>HHD_dep_housing_PC</th>
      <th>Polygon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E00053353</td>
      <td>1000.455806</td>
      <td>0.282787</td>
      <td>0.114754</td>
      <td>0.336066</td>
      <td>0.077869</td>
      <td>POLYGON ((415817.093 440872.597, 415821.094 44...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E00053354</td>
      <td>771.145845</td>
      <td>0.172932</td>
      <td>0.067669</td>
      <td>0.308271</td>
      <td>0.052632</td>
      <td>POLYGON ((415078.000 439967.001, 415058.323 43...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E00053355</td>
      <td>418.265674</td>
      <td>0.112000</td>
      <td>0.048780</td>
      <td>0.219512</td>
      <td>0.016000</td>
      <td>POLYGON ((416252.367 439816.041, 416253.270 43...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E00053356</td>
      <td>194.292561</td>
      <td>0.215278</td>
      <td>0.041667</td>
      <td>0.340278</td>
      <td>0.048611</td>
      <td>POLYGON ((416668.000 439392.028, 416667.653 43...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E00053357</td>
      <td>432.352481</td>
      <td>0.141844</td>
      <td>0.034965</td>
      <td>0.212766</td>
      <td>0.007092</td>
      <td>POLYGON ((415143.909 439176.235, 415143.000 43...</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_df.to_file('Wrangled dataframes/final_df.geojson', driver='GeoJSON')
```
