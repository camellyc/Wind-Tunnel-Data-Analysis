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
    dt = preprocessing_params['dt']

    xmin = preprocessing_params['xmin']
    xmax = preprocessing_params['xmax']
    ymin = preprocessing_params['ymin']
    ymax = preprocessing_params['ymax']
    zmin = preprocessing_params['zmin']
    zmax = preprocessing_params['zmax']

    frames_before = preprocessing_params['frames_before']
    frames_after = preprocessing_params['frames_after']
    frames_after_min = preprocessing_params['frames_after_min']

    preprocessed_data_subdir =  preprocessing_params['preprocessed_data_subdir']
    preprocessed_data_fname_suffix = preprocessing_params['only_triggering_trajecs']

    # filename for saving trajectory stats
    trajec_stats_yaml_filename = preprocessing_params['trajec_stats_yaml_filename']

    filtered_aligned_events = preprocessing_params['filtered_aligned_events']


    trigger_yaml_subdir = preprocessing_params['trigger_yaml_subdir']
    trigger_yaml_suffix = preprocessing_params['trigger_yaml_suffix']
    trigger_column_names_key = preprocessing_params['trigger_column_names_key']
    trigger_bag_name_identifier = preprocessing_params['trigger_bag_name_identifier']
    trigger_first_column_to_rename = preprocessing_params['trigger_first_column_to_rename']
    ##########################################################################################################

    print("")
    print("----------------------------------------------------")
    print("Loading from directory: ")
    print(data_directory)

    # Load preprocessed data
    preprocessed_directory = os.path.join(data_directory, preprocessed_data_subdir)
    preprocessed_data_filename = braid_filemanager.get_filename(preprocessed_directory, 
                                                                preprocessed_data_fname_suffix)
    print(preprocessed_data_filename)
    stamped_df = pd.read_hdf(preprocessed_data_filename)

    # load and cleanup trigger data
    trigger_df, trigger_column_names = optotrigger_pandas_functions.load_and_cleanup_trigger_df(data_directory, preprocessing_params)

    
    # unique objects
    objids = stamped_df.obj_id.unique()
    
    print('Extracting individual events, filling missing data with nans')
    print('Min frames before: ' + str(frames_before))
    print('Min frames after: ' + str(frames_after_min))
    print('Filling to total frames after: ' + str(frames_after))
    # pull out events -- there might be multiple events for a single object if the trajectory is very long
    trajec_dfs = []
    event_number = 0
    for objid in objids:
        trajec_slice = None
        trajec = stamped_df[stamped_df.obj_id==objid].copy()
        trajec.loc[:,'timestamp'] = trajec.timestamp.interpolate()

        # all the flash frames for a given trajectory
        flash_frames = trajec[trajec.flash_frame==True].frame.to_numpy()
        for flash_frame in flash_frames:
            # make sure this specific obj id and flash frame are in the trigger df
            if len(trigger_df.query("flash_frame == " + str(flash_frame) + " and obj_id == " + str(objid))) == 0:
                trajec_slice = None
            else:
                if flash_frame - trajec.frame.min() > 20:
                    if trajec.frame.max() - flash_frame > frames_after_min:
                        trajec_slice = trajec[(trajec.frame>=flash_frame-frames_before)*(trajec.frame<=flash_frame+frames_after)].copy()

                        # don't accept events with multiple flashes within the analysis window
                        if trajec_slice.flash_frame.sum() > 1:
                            frame_diff = np.diff(trajec_slice[trajec_slice.flash_frame==1].frame.values)[0]
                            print('obj id has more than 1 flash in time interval, skipping: ' + str(trajec_slice.obj_id.median()) + ' num frames between flashes: ' + str(frame_diff))
                            trajec_slice = None

            if trajec_slice is not None:
                frames_to_add = frames_after - (trajec_slice.frame.max()-flash_frame)

                # add blank frames to fill empty rows
                if frames_to_add > 0:
                    nan = np.ones(frames_to_add)*float('nan')
                    frames = np.arange(trajec_slice.frame.max()+1, trajec_slice.frame.max()+frames_to_add+1)
                    timestamps = (frames-frames[0]+1)*dt + trajec_slice.timestamp.max()
                    obj_id = np.array([trajec_slice.obj_id.values[0]]*len(nan))
                    obj_id_unique = np.array([trajec_slice.obj_id_unique.values[0]]*len(nan)).astype(str)
                    blank_rows = {key: nan for key in trajec_slice.keys()}
                    blank_rows['frame'] = frames
                    blank_rows['flash_frame'] = False
                    blank_rows['timestamp'] = timestamps
                    blank_rows['obj_id'] = obj_id
                    blank_rows['obj_id_unique'] = obj_id_unique
                    blank_rows['P00'] = np.zeros_like(nan)
                    blank_rows['P11'] = np.zeros_like(nan)
                    blank_rows['P22'] = np.zeros_like(nan)
                    blank_rows['P33'] = np.zeros_like(nan)
                    blank_rows['P44'] = np.zeros_like(nan)
                    blank_rows['P55'] = np.zeros_like(nan)
                    blank_rows = pd.DataFrame(blank_rows)
                    trajec_slice = pd.concat([trajec_slice, blank_rows], ignore_index=True)

                # flash stuff -- helps with slicing later
                for col in trigger_column_names:
                    trajec_slice[col] = trajec_slice[~trajec_slice[col].isna()][col].values[0]

                # time relative to flash -- and clean it up
                time_at_flash = trajec_slice[trajec_slice.flash_frame==True].timestamp.to_numpy()[0]
                trajec_slice['time_relative_to_flash'] = trajec_slice.timestamp - time_at_flash
                trajec_slice['time_relative_to_flash'] = np.round(trajec_slice['time_relative_to_flash'].values/dt).astype(int)*dt

                trajec_slice['obj_id_unique_event'] = str(trajec_slice.obj_id_unique.values[0]) + '_' + str(event_number)
                event_number += 1

                trajec_dfs.append(trajec_slice)

    # merge all the events
    stamped_df_aligned = pd.concat(trajec_dfs, ignore_index=True)

    print('Calculating angular velocities')
    stamped_df_aligned = flymath.assign_course_and_ang_vel_to_dataframe(stamped_df_aligned, do_cvx_smoother=False, object_key='obj_id_unique_event')

    # save some stats
    num_trajecs = len(stamped_df_aligned.obj_id.unique())
    num_triggers = sum( stamped_df_aligned['flash_frame'].values )
    stats_events = {'4_trigger_events_aligned_filtered': 
                          {'num_trajecs': {'text': "Filtered number of trajecs that triggered: ",
                                          'number': int(num_trajecs),
                                          },
                          'num_triggers': {'text': "Filtered number of trigger events: ",
                                           'number': int(num_triggers),
                                           },
                           'filtering_protocol': {'frames_before': frames_before,
                                                 'frames_after': frames_after,
                                                 'frames_after_min': frames_after_min,
                                                 },
                           'data_filename': None,}
                     }
    # print stats
    print("----------------------------------------------------")
    print("Only trajectories that triggered stats: ")
    print( stats_events['4_trigger_events_aligned_filtered']['num_trajecs']['text'] + str(stats_events['4_trigger_events_aligned_filtered']['num_trajecs']['number']))
    print( stats_events['4_trigger_events_aligned_filtered']['num_triggers']['text'] + str(stats_events['4_trigger_events_aligned_filtered']['num_triggers']['number']))
    print("----------------------------------------------------")

    
    # filter based on volume -- if trajecs leave specified volume, ignore them
    print('Filtering based on volume')
    print('xmin: ' + str(xmin))
    print('xmax: ' + str(xmax))
    print('ymin: ' + str(ymin))
    print('ymax: ' + str(ymax))
    print('zmin: ' + str(zmin))
    print('zmax: ' + str(zmax))
    obj_id_key='obj_id_unique_event'
    obj_ids = braid_slicing.get_obj_ids_that_stay_in_volume(stamped_df_aligned, obj_id_key=obj_id_key,
                                    xmin = xmin,
                                    xmax = xmax,
                                    ymin = ymin,
                                    ymax = ymax,
                                    zmin = zmin,
                                    zmax = zmax)
    stamped_df_aligned_trimmed = stamped_df_aligned[stamped_df_aligned[obj_id_key].isin(obj_ids)]
    
    # save the trimmed data
    fname = braid_filemanager.save_preprocessed_braidz(data_directory, stamped_df_aligned_trimmed, 
                                 suffix=filtered_aligned_events)
    
    # save some stats
    num_trajecs = len(stamped_df_aligned_trimmed.obj_id.unique())
    num_triggers = len( stamped_df_aligned_trimmed['obj_id_unique_event'].unique() )
    stats_events_volume_filtered = {'5_trigger_events_aligned_filtered_volume_trimmed': 
                          {'num_trajecs': {'text': "Filtered number of trajecs that triggered and stayed in volume: ",
                                          'number': int(num_trajecs),
                                          },
                          'num_triggers': {'text': "Filtered number of trigger events that stayed in volume: ",
                                           'number': int(num_triggers),
                                           },
                           'filtering_protocol': {'frames_before': frames_before,
                                                 'frames_after': frames_after,
                                                 'frames_after_min': frames_after_min,
                                                 'xmin': xmin,
                                                 'xmax': xmax,
                                                  'ymin': ymin,
                                                  'ymax': ymax,
                                                  'zmin': zmin,
                                                  'zmax': zmax,
                                                 },
                           'data_filename': fname,}
                     }
    # print stats
    print("----------------------------------------------------")
    print("Only trajectories that triggered and stayed in volume: ")
    print( stats_events_volume_filtered['5_trigger_events_aligned_filtered_volume_trimmed']['num_trajecs']['text'] + str(stats_events_volume_filtered['5_trigger_events_aligned_filtered_volume_trimmed']['num_trajecs']['number']))
    print( stats_events_volume_filtered['5_trigger_events_aligned_filtered_volume_trimmed']['num_triggers']['text'] + str(stats_events_volume_filtered['5_trigger_events_aligned_filtered_volume_trimmed']['num_triggers']['number']))
    print("----------------------------------------------------")

    
    # save the filtering and stats info
    preprocessed_data_dir = os.path.join(data_directory, preprocessed_data_subdir)
    stats_yaml_fname = os.path.join(preprocessed_data_dir, trajec_stats_yaml_filename)
    with open(stats_yaml_fname) as stream:
        all_stats = yaml.safe_load(stream)
    try:
        all_stats = {**all_stats, **stats_events, **stats_events_volume_filtered}
    except:
        all_stats = stats_events
    with open(stats_yaml_fname, 'w') as yaml_file:
        yaml.dump(all_stats, yaml_file, default_flow_style=False)
