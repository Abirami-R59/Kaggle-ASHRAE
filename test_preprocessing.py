# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 20:45:25 2019

@author: ar1
"""
import gc
import time
import numpy as np 
import pandas as pd
from multiprocessing import Pool
from functools import partial

import os
for dirname, _, filenames in os.walk('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
## Function to read the data sets
def read_data(path):
    """
    Reads the data from the path and 
    prints the shape and sample rows of the dataframe
    """
    df = pd.read_csv(path)
    print(df.shape)
    print(df.head())
    return df

def parse_timestamp(df, col):
    """
    Converts the timestamp column from object to datetime type
    """
    df[col] = pd.to_datetime(df[col])
    return df

def merge_datasets(train, weather_train, building_metadata):
    """
    Merges the 3 dataframes train, weather_train and building_metadata
    to a single dataframe and removes the duplicate rows if any
    """
    print("Merging weather data with building metadata")
    weather_build_meta = pd.merge(weather_train, building_metadata, on='site_id', how='left')
    print(weather_build_meta.shape)
    
    print("Dropping duplicates, if any")
    weather_build_meta = weather_build_meta.drop_duplicates()
    print(weather_build_meta.shape)
    
    print("Merging with train data")
    final_data = pd.merge(weather_build_meta, train, on = ['building_id', 'timestamp'], how='outer')
    print(final_data.shape)
    
    print("Dropping duplicates, if any")
    final_data = final_data.drop_duplicates()
    print(final_data.shape)
    
    return final_data

def parallelize_dataframe(df, group_list, func, **kwargs):
    """
    Parallelize the dataframe based on groups in a grouped dataframe, 
    runs the specified function on each of the groups in parallel and
    combines it to a dataframe
    """
    df_split = df.groupby(group_list)
    pool = Pool(os.cpu_count())
#    if args is not None:
#        func_x=partial(func, **kwargs)
#    else:
    func_x = partial(func, **kwargs)
    ret_list = pool.map(func_x, [group for name, group in df_split])
    df = pd.DataFrame()
    print("for loop")
    for dat in ret_list:
        df = df.append(dat, ignore_index=True)
    pool.close()
    pool.join()
    return df

def fill_attributes(df, attr_list):
    print(df['building_id'].unique()[0])
    for attr in attr_list:
        print(attr)
        attr_value_list = df[attr].dropna().unique().tolist()
        if len(attr_value_list) == 1:
            df[attr] = df[attr].fillna(attr_value_list[0])
        else:
            print("No unique values to fill the attributes")
    return df

def fill_numerical_data(df):
    print("Forward Fill")
    df[['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = df.groupby(['building_id','meter'])['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed'].fillna(method='ffill')
    
    print("Backward Fill")
    df[['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = df.groupby(['building_id','meter'])['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed'].fillna(method='bfill')
    
    return df
    

def create_features(df, limit_sqft):
    print(df['building_id'].unique()[0])
    ## Features from timestamp
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    
    df['age'] = df['year'] - df['year_built']
    
    df['square_feet_profile'] = np.where((df['square_feet'] >= limit_sqft['min']) & (df['square_feet'] < limit_sqft['25%']), 'small',
                                         np.where((df['square_feet'] >= limit_sqft['25%']) & (df['square_feet'] < limit_sqft['50%']), 'medium',
                                         np.where((df['square_feet'] >= limit_sqft['50%']) & (df['square_feet'] < limit_sqft['75%']), 'big', 
                                         np.where((df['square_feet'] >= limit_sqft['75%']) & (df['square_feet'] < limit_sqft['max']), 'huge', np.nan))))
    
    return df


def sine_transform(input_df, normalize_var):
    """Transform a input DF to Sine transform"""
    col_names = input_df.columns
    if len(col_names) == 1:
        transformed_array = np.sin(2 * np.pi * input_df[col_names[0]] / normalize_var)
        transformed_df = pd.DataFrame({f'sine_{col_names[0]}': transformed_array})
    else:
        transformed_df = None
    return transformed_df

def cosine_transform(input_df, normalize_var):
    """Transform a input DF to Cosine transform"""
    col_names = input_df.columns
    if len(col_names) == 1:
        transformed_array = np.cos(2 * np.pi * input_df[col_names[0]] / normalize_var)
        transformed_df = pd.DataFrame({f'cosine_{col_names[0]}': transformed_array})
    else:
        transformed_df = None
    return transformed_df

def create_sin_cos_features(df):
    print(df['building_id'].unique()[0])
    df['month_sin'] = sine_transform(df[['month']], 12)
    df['month_cos'] = cosine_transform(df[['month']], 12)

    df['day_sin'] = sine_transform(df[['day']], 31)
    df['day_cos'] = cosine_transform(df[['day']], 31)

    df['hour_sin'] = sine_transform(df[['hour']], 23)
    df['hour_cos'] = cosine_transform(df[['hour']], 23)

    df['wind_dir_sin'] = sine_transform(df[['wind_direction']], 360)
    df['wind_dir_cos'] = cosine_transform(df[['wind_direction']], 360)
    
    return df
    


def main():
    start = time.time()
    #global meter_grp
    ## Reading Training Data
    test = read_data('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/input/test.csv')
    weather_test = read_data('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/input/weather_test.csv')
    building_metadata = read_data('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/input/building_metadata.csv')
    
    ## Parsing Timestamps
    test = parse_timestamp(test, 'timestamp')
    weather_test = parse_timestamp(weather_test, 'timestamp')
    
    test_final = merge_datasets(test, weather_test, building_metadata)
    print(test_final.head())
    print(test_final.columns)
    print(test_final.shape)
    
    del test, weather_test, building_metadata
    print(gc.collect())
    
    for meter in test_final.meter.unique():
        print ("Meter " + str(meter))
        test_final_0 = test_final[test_final['meter'] == meter].reset_index(drop=True)
        print(test_final_0.shape)
        
        print("Filling attribute data")
        test_final_0[['building_id', 'site_id', 'primary_use', 'square_feet', 'year_built']] = parallelize_dataframe(test_final_0[['building_id', 'site_id', 'primary_use', 'square_feet', 'year_built']], ['building_id'], fill_attributes, attr_list=['site_id', 'primary_use', 'square_feet', 'year_built'])
        print(test_final_0.isnull().sum())
        
    
        ## Filling Numerical data
        print("Filling numerical data")
        
        test_final_0[['building_id', 'meter', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
           'wind_direction', 'wind_speed']] = parallelize_dataframe(test_final_0[['building_id', 'meter', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
           'wind_direction', 'wind_speed']], ['building_id','meter'], fill_numerical_data)
           
        print(test_final_0.isnull().sum())
        
    #    print("Forward Fill")
    #    test_final_0[['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
    #       'wind_direction', 'wind_speed']] = test_final_0.groupby(['building_id','meter'])['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
    #       'wind_direction', 'wind_speed'].fillna(method='ffill')
    #    print(test_final_0.isnull().sum())
    #    
    #    print("Backward Fill")
    #    test_final_0[['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
    #       'wind_direction', 'wind_speed']] = test_final_0.groupby(['building_id','meter'])['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
    #       'wind_direction', 'wind_speed'].fillna(method='bfill')
    #    print(test_final_0.isnull().sum())
    #    
    
        ## Creating Features
        print("Creating features")
        #global limit_sqft
        limit_sqft = test_final['square_feet'].describe()
        test_final_0 = parallelize_dataframe(test_final_0, ['building_id'], create_features, limit_sqft=limit_sqft)
        print(test_final_0.head())
        
        print("Creating cyclical features")
        test_final_0 = parallelize_dataframe(test_final_0, ['building_id'], create_sin_cos_features)
        print(test_final_0.head())
        
        test_final_0.to_csv('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction\output/test_final_'+str(meter)+'.csv', index=False)

    
    end = time.time()
    print(end-start)
    
    
    return test_final_0


if __name__ == "__main__":
    msg = main()
    print(msg)
    
    