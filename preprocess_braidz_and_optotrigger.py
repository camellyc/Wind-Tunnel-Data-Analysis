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
    braid_df = braid_df[0:preprocessing_params['last_frame']]
    
    # load and cleanup trigger data
    trigger_df, trigger_column_names = optotrigger_pandas_functions.load_and_cleanup_trigger_df(data_directory, preprocessing_params)

    # save some stats
    num_trajecs = len(braid_df.obj_id.unique())
    num_triggers = len(trigger_df)
    stats_raw = {'1_raw_braidz': 
                      {'num_trajecs': {'text': "Number of trajecs before filtering: ",
                                      'number': int(num_trajecs),},
                      'num_triggers': {'text': "Number of triggers total: ",
                                       'number': int(num_triggers),
                                       },}
                 }
    # print stats
    print("----------------------------------------------------")
    print("Raw data stats: ")
    print( stats_raw['1_raw_braidz']['num_trajecs']['text'] + str(stats_raw['1_raw_braidz']['num_trajecs']['number']))
    print( stats_raw['1_raw_braidz']['num_triggers']['text'] + str(stats_raw['1_raw_braidz']['num_triggers']['number']))
    print("----------------------------------------------------")
    
    # do some basic filtering of object length and xdist travelled
    print('Filtering for minimum trajec length and xdistance travelled')
    long_obj_ids = braid_slicing.get_long_obj_ids_fast_pandas(braid_df, length=min_length)
    stamped_df_culled = braid_slicing.get_data_frame_slice_from_obj_ids(braid_df, long_obj_ids)
    long_xdist_objids = braid_slicing.get_trajectories_that_travel_far(stamped_df_culled, 
                                                                       xdist_travelled=min_xdist_travelled)
    stamped_df_culled = braid_slicing.get_data_frame_slice_from_obj_ids(stamped_df_culled, long_xdist_objids)
    print('After basic filtering: ' + str(len(stamped_df_culled.obj_id.unique())) + ' trajectories left')

    # add trigger data to braid dataframe
    stamped_df_culled = optotrigger_pandas_functions.assign_single_flash_values(stamped_df_culled, trigger_df, trigger_column_names)
    stamped_df_culled = lights_on_function(stamped_df_culled, trigger_df, value_to_assign=1, dt=dt)
    #stamped_df_culled = optotrigger_pandas_functions.assign_time_since_lights_on(stamped_df_culled) <<< this is way to slow! instead of searching through the braid_df for when lights are on, work from the trigger file. First construct the lights on pattern for each flash, then use that to find last time lights were on.

    # assign unique obj id
    name = os.path.basename(braidz_filename).split('.')[0]
    stamped_df_culled = braid_slicing.assign_unique_id(stamped_df_culled, name)

    # calculate ground speed:
    print('Calculating speed (xy ground speed)')
    stamped_df_culled['speed_xy'] = np.sqrt(stamped_df_culled.xvel**2 + stamped_df_culled.yvel**2)

    # calculate angular velocities -- this is very slow, better to do in trimming protocol
    #print('Calculating angular velocities')
    #stamped_df_culled = flymath.assign_course_and_ang_vel_to_dataframe(stamped_df_culled, do_cvx_smoother=False)
    
    # save the preprocessed data
    fname = braid_filemanager.save_preprocessed_braidz(data_directory, stamped_df_culled, suffix=slightly_filtered_data)
      
    # save some stats
    num_trajecs = len(stamped_df_culled.obj_id.unique())
    d = braid_df[braid_df['obj_id'].isin(trigger_df['obj_id'])==True]
    num_triggers = sum( d['frame'].isin(trigger_df['flash_frame'].to_list()) )
    stats_slightly_filtered = {'2_filtered_trajecs': 
                      {'num_trajecs': {'text': "Filtered number of trajecs: ",
                                      'number': int(num_trajecs),
                                      },
                      'num_triggers': {'text': "Filtered number of trigger events: ",
                                       'number': int(num_triggers),
                                       },
                      'filtering_protocol': {'min_length': min_length,
                                             'min_xdist_travelled': min_xdist_travelled},
                      'data_filename': fname}
                 }
    # print stats
    print("----------------------------------------------------")
    print("Slightly filtered stats: ")
    print( stats_slightly_filtered['2_filtered_trajecs']['num_trajecs']['text'] + str(stats_slightly_filtered['2_filtered_trajecs']['num_trajecs']['number']))
    print( stats_slightly_filtered['2_filtered_trajecs']['num_triggers']['text'] + str(stats_slightly_filtered['2_filtered_trajecs']['num_triggers']['number']))
    print("----------------------------------------------------")
        
    # only keep trajectories that triggered
    stamped_df_culled = stamped_df_culled[stamped_df_culled.obj_id.isin(trigger_df.obj_id)==True].copy()

    # save the optotrigger data
    fname = braid_filemanager.save_preprocessed_braidz(data_directory, stamped_df_culled, suffix=only_triggering_trajecs)
    
    # save some stats
    num_trajecs = len(stamped_df_culled.obj_id.unique())
    num_triggers = sum( stamped_df_culled['flash_frame'].values )
    stats_triggers_only = {'3_triggered_trajecs': 
                          {'num_trajecs': {'text': "Filtered number of trajecs that triggered: ",
                                          'number': int(num_trajecs),
                                          },
                          'num_triggers': {'text': "Filtered number of trigger events: ",
                                           'number': int(num_triggers),
                                           },
                           'filtering_protocol': 'triggered trajecs only',
                           'data_filename': fname,}
                     }
    # print stats
    print("----------------------------------------------------")
    print("Only trajectories that triggered stats: ")
    print( stats_triggers_only['3_triggered_trajecs']['num_trajecs']['text'] + str(stats_triggers_only['3_triggered_trajecs']['num_trajecs']['number']))
    print( stats_triggers_only['3_triggered_trajecs']['num_triggers']['text'] + str(stats_triggers_only['3_triggered_trajecs']['num_triggers']['number']))
    print("----------------------------------------------------")
    
    # save the stats to a yaml file
    all_stats = {**stats_raw, **stats_slightly_filtered, **stats_triggers_only} # joins all stats dictionaries
    preprocessed_data_directory = os.path.join(data_directory, preprocessed_data_subdir)
    stats_yaml = os.path.join(preprocessed_data_directory, trajec_stats_yaml_filename)
    with open(stats_yaml, 'w') as yaml_file:
        yaml.dump(all_stats, yaml_file, default_flow_style=False)
        
    # instructions on how to load stats
    if 0:
        with open(stats_yaml) as stream:
            stats_yaml = yaml.safe_load(stream)
        print(stats_yaml)
