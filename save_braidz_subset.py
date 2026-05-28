import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import argparse

from braid_analysis import braid_filemanager
from braid_analysis import braid_slicing
from braid_analysis import braid_analysis_plots
from braid_analysis import flymath

from optotrigger_analysis import optotrigger_pandas_functions
from optotrigger_analysis import bag2hdf5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help="data directory")
    parser.add_argument('params', type=str, help="full path to yaml file with preprocessing parameters")
    args = parser.parse_args()
    
    data_directory = args.directory
    preprocessing_parameters_yaml_fname = args.params

    ##########################################################################################################
    # load preprocessing parameters
    with open(preprocessing_parameters_yaml_fname) as stream:
        preprocessing_params = yaml.safe_load(stream)

    # extract preprocessing params 
    preprocessed_data_subdir = preprocessing_params['preprocessed_data_subdir']
    
    dt = preprocessing_params['dt']


    trigger_yaml_subdir = preprocessing_params['trigger_yaml_subdir']
    trigger_yaml_suffix = preprocessing_params['trigger_yaml_suffix']
    trigger_column_names_key = preprocessing_params['trigger_column_names_key']
    trigger_bag_name_identifier = preprocessing_params['trigger_bag_name_identifier']
    trigger_first_column_to_rename = preprocessing_params['trigger_first_column_to_rename']

    min_length = preprocessing_params['min_length']
    min_xdist_travelled = preprocessing_params['min_xdist_travelled']

    # filename for saving trajectory stats
    trajec_stats_yaml_filename = preprocessing_params['trajec_stats_yaml_filename']
    
    # preprocessed braid suffix
    slightly_filtered_data = preprocessing_params['slightly_filtered_data']
    only_triggering_trajecs = preprocessing_params['only_triggering_trajecs']

    # this function depends on the experiment
    lights_on_function = optotrigger_pandas_functions.assign_single_flash_lights_on

    ##########################################################################################################
    
    if braid_filemanager.preprocessed_braidz_exists(data_directory, sub_directory=preprocessed_data_subdir, suffix=slightly_filtered_data):
        print("----------------------------------------------------")
        print('WARNING: preprocessed data already exists.. exiting.')
        print('Clear out subdirectory and try running again')
        print('Subdir: ' + preprocessed_data_subdir )
        print("----------------------------------------------------")
        sys.exit(0)

    print("")
    print("----------------------------------------------------")
    print("Loading from directory: ")
    print(data_directory)
    
    # load braid as pandas dataframe
    braidz_filename = braid_filemanager.get_filename(data_directory, '.braidz')
    print(braidz_filename)
    print("")
    braid_df = braid_filemanager.load_filename_as_dataframe_3d(braidz_filename)
    braid_df = braid_df[0:20000]

    braid_df.to_hdf('braidz_subset.hdf', 'braidz_subset')