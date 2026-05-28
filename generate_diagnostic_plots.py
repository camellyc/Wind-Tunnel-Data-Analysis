import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import argparse

import figurefirst as fifi
import cairosvg

from braid_analysis import braid_filemanager
from braid_analysis import braid_slicing
from braid_analysis import braid_analysis_plots
from braid_analysis import flymath
from optotrigger_analysis import optotrigger_analysis_plots

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help="data directory")
    args = parser.parse_args()
    
    data_directory = args.directory
    fifi_template = 'diagnostic_figure_template.svg'

    # time between frames (could be extracted from data instead)
    dt = 0.01

    # dimensions of wind tunnel
    xmin = -0.75
    xmax = 0.75
    ymin = -0.3
    ymax = 0.3
    zmin = 0
    zmax = 0.6

    # one pdf will be generated for each value in this column
    trigger_column_name = 'flash_duration'

    # for rastering
    ZORDER = 10

    # which dataset to open
    preprocessed_data_subdir = 'preprocessed_data'
    preprocessed_data_fname_suffix = 'preprocessed_optotrigger_trimmed.hdf'
    trajectory_stats_yaml_fname = 'trajectory_statistics_and_filtering.yaml'

    # load data
    preprocessed_directory = os.path.join(data_directory, preprocessed_data_subdir)
    preprocessed_data_filename = braid_filemanager.get_filename(preprocessed_directory, 
                                                                preprocessed_data_fname_suffix)
    print('Loading: ')
    print(preprocessed_data_filename)
    df_3d = pd.read_hdf(preprocessed_data_filename)

    # load statistics
    full_trajectory_stats_yaml_fname = os.path.join(data_directory, preprocessed_data_subdir, trajectory_stats_yaml_fname)
    with open( full_trajectory_stats_yaml_fname) as stream:
        stats_yaml = yaml.safe_load(stream)

    # get trigger value options
    trigger_values = df_3d[trigger_column_name].unique()

    for trigger_value in trigger_values:

        trigger_value_str = str(trigger_value)
        trigger_value_str = trigger_value_str.replace('.', '_')
        output_figure_name = os.path.join(data_directory, 'diagnostic_plots_' + trigger_column_name + '_' + trigger_value_str + '.svg')
        layout = fifi.svg_to_axes.FigureLayout(fifi_template, autogenlayers=True, make_mplfigures=True, hide_layers=[])

        ###################################################################################################
        # write statistics
        layout.svgitems['filename'].text = os.path.basename(preprocessed_data_filename)
        layout.svgitems['filename'].style['font-size'] = 2
        layout.svgitems['trigger_condition'].text = trigger_column_name + ': ' + str(trigger_value)
        layout.svgitems['trigger_condition'].style['font-size'] = 2
        layout.svgitems['experiment_info'].text = data_directory
        layout.svgitems['experiment_info'].style['font-size'] = 2
        for i, key in enumerate(stats_yaml.keys()):
            layout.svgitems['statistics_line_'+str(i+1)].text = stats_yaml[key]['num_trajecs']['text'] + ' ' + str(stats_yaml[key]['num_trajecs']['number'])
            layout.svgitems['statistics_line_'+str(i+1)].style['font-size'] = 2
        layout.apply_svg_attrs()

        ###################################################################################################
        # get slice for trigger value
        df_3d_slice = df_3d[df_3d[trigger_column_name]==trigger_value].copy()

        tmax = df_3d_slice[df_3d_slice.x.notna()].time_relative_to_flash.max()

        ###################################################################################################
        # speed and angular velocity histograms
        ax = layout.axes[('speed_histogram', 'speed_histogram')]
        braid_analysis_plots.plot_speed_xy_histogram(df_3d_slice, ax=ax)
        ax.set_rasterization_zorder(ZORDER)
        fifi.mpl_functions.set_fontsize(ax, 6)
        layout.append_figure_to_layer(layout.figures['speed_histogram'], 'speed_histogram', cleartarget=True)

        ax = layout.axes[('angvel_histogram', 'angvel_histogram')]
        braid_analysis_plots.plot_speed_xy_histogram(df_3d_slice, ax=ax, speed_key='ang_vel_smoothish', bins=np.arange(-10, 10, 0.2))
        ax.set_rasterization_zorder(ZORDER)
        fifi.mpl_functions.set_fontsize(ax, 6)
        layout.append_figure_to_layer(layout.figures['angvel_histogram'], 'angvel_histogram', cleartarget=True)

        ###################################################################################################
        # starting and ending points
        ax_xz = layout.axes[('starts_and_stops', 'xz')]
        ax_xy = layout.axes[('starts_and_stops', 'xy')]
        obj_id_key = 'obj_id_unique_event'
        braid_analysis_plots.plot_starting_and_ending_points(df_3d_slice, obj_id_key,
                                                             xmin, 
                                                             xmax, 
                                                             ymin, 
                                                             ymax, 
                                                             zmin, 
                                                             zmax, 
                                                             start_or_end='start',
                                                             ax_xy=ax_xy,
                                                             ax_xz=ax_xz,
                                                             dot_size=1)

        braid_analysis_plots.plot_starting_and_ending_points(df_3d_slice, obj_id_key,
                                                             xmin, 
                                                             xmax, 
                                                             ymin, 
                                                             ymax, 
                                                             zmin, 
                                                             zmax, 
                                                             start_or_end='end',
                                                             ax_xy=ax_xy,
                                                             ax_xz=ax_xz,
                                                             dot_size=1)
        ax_xy.set_rasterization_zorder(ZORDER)
        ax_xz.set_rasterization_zorder(ZORDER)
        fifi.mpl_functions.set_fontsize(ax_xy, 6)
        fifi.mpl_functions.set_fontsize(ax_xz, 6)
        layout.append_figure_to_layer(layout.figures['starts_and_stops'], 'starts_and_stops', cleartarget=True)

        ###################################################################################################
        # flash locations
        ax_xz = layout.axes[('flash_locations', 'xz')]
        ax_xy = layout.axes[('flash_locations', 'xy')]
        optotrigger_analysis_plots.plot_trigger_locations(df_3d_slice, 
                                                          xmin, xmax, ymin, ymax, zmin, zmax, 
                                                          key='flash_frame',
                                                          ax_xy=ax_xy,
                                                          ax_xz=ax_xz,
                                                          dot_size=1)
        ax_xy.set_rasterization_zorder(ZORDER)
        ax_xz.set_rasterization_zorder(ZORDER)
        fifi.mpl_functions.set_fontsize(ax_xy, 6)
        fifi.mpl_functions.set_fontsize(ax_xz, 6)
        layout.append_figure_to_layer(layout.figures['flash_locations'], 'flash_locations', cleartarget=True)

        ###################################################################################################
        # occupancy maps
        ax_xz = layout.axes[('occupancy', 'xz')]
        ax_xy = layout.axes[('occupancy', 'xy')]
        braid_analysis_plots.plot_occupancy_heatmaps(df_3d_slice, 
                                                         xmin, xmax, ymin, ymax, zmin, zmax, 
                                                         resolution=0.01,
                                                          ax_xy=ax_xy,
                                                          ax_xz=ax_xz)
        ax_xy.set_rasterization_zorder(ZORDER)
        ax_xz.set_rasterization_zorder(ZORDER)
        fifi.mpl_functions.set_fontsize(ax_xy, 6)
        fifi.mpl_functions.set_fontsize(ax_xz, 6)
        layout.append_figure_to_layer(layout.figures['occupancy'], 'occupancy', cleartarget=True)


        ###################################################################################################
        # individual trajectories
        obj_ids = df_3d_slice.obj_id_unique_event.unique()
        num_trajecs_to_plot = len(obj_ids)
        if num_trajecs_to_plot > 8:
            num_trajecs_to_plot = 8
        obj_ids= np.random.choice(obj_ids, num_trajecs_to_plot, replace=False)

        for i, obj_id in enumerate(obj_ids):
            ax_xz = layout.axes[('fly_'+str(i+1), 'xz')]
            ax_xy = layout.axes[('fly_'+str(i+1), 'xy')]
            
            ax_xz.set_aspect('equal')
            ax_xy.set_aspect('equal')
            
            trajec = df_3d_slice[df_3d_slice.obj_id_unique_event==obj_id]
            braid_analysis_plots.plot_xy_trajectory_with_color_overlay(trajec, 
                                                  obj_id_key='obj_id_unique_event',
                                                  column_for_color='lights_on',
                                                  plane = 'xz', # xy or xz or yz
                                                  cmap='seismic', 
                                                  vmin=-2, vmax=2,
                                                  dot_size=1,
                                                  ax=ax_xz)
            braid_analysis_plots.plot_xy_trajectory_with_color_overlay(trajec, 
                                                  obj_id_key='obj_id_unique_event',
                                                  column_for_color='lights_on',
                                                  plane = 'xy', # xy or xz or yz
                                                  cmap='seismic', 
                                                  vmin=-2, vmax=2,
                                                  dot_size=1,
                                                  ax=ax_xy)

            ax_xy.set_rasterization_zorder(ZORDER)
            ax_xz.set_rasterization_zorder(ZORDER)
            fifi.mpl_functions.set_fontsize(ax_xy, 6)
            fifi.mpl_functions.set_fontsize(ax_xz, 6)

            ax_xy.axis('off')
            ax_xz.axis('off')

            layout.append_figure_to_layer(layout.figures['fly_'+str(i+1)], 'fly_'+str(i+1), cleartarget=True)

        ###################################################################################################
        # course direction
        ax = layout.axes[('course_vs_time', 'course_vs_time')]
        braid_analysis_plots.plot_column_vs_time(df_3d_slice,
                                                    column='course_smoothish',
                                                    time_key='time_relative_to_flash',
                                                    norm_columns_to_min_max=True,
                                                    cmap='bone_r',
                                                    vmin=0, vmax=1,
                                                    bin_y= np.arange(-np.pi, np.pi+0.05, 0.05),
                                                    bin_x= np.arange(-0.5+dt/2, tmax+dt+dt/2, dt),
                                                    ax=ax  )

        if trigger_value != 0:
            ax.fill_betweenx([-np.pi, np.pi], 0, trigger_value/1000., facecolor='red', edgecolor='none', alpha=0.3)
        else:
            ax.fill_betweenx([-np.pi, np.pi], 0, trigger_value/1000., facecolor='gray', edgecolor='none', alpha=0.3)

        ax.set_rasterization_zorder(ZORDER)
        fifi.mpl_functions.set_fontsize(ax, 6)

        layout.append_figure_to_layer(layout.figures['course_vs_time'], 'course_vs_time', cleartarget=True)


        ###################################################################################################
        # speed vs time
        ax = layout.axes[('speed_vs_time', 'speed_vs_time')]
        braid_analysis_plots.plot_column_vs_time(df_3d_slice,
                                                    column='speed_xy',
                                                    time_key='time_relative_to_flash',
                                                    norm_columns_to_min_max=True,
                                                    cmap='bone_r',
                                                    vmin=0, vmax=1,
                                                    bin_y= np.arange(0, 0.6+0.01, 0.01),
                                                    bin_x= np.arange(-0.5+dt/2, tmax+dt+dt/2, dt),
                                                    ax=ax  )

        if trigger_value != 0:
            ax.fill_betweenx([-np.pi, np.pi], 0, trigger_value/1000., facecolor='red', edgecolor='none', alpha=0.3)
        else:
            ax.fill_betweenx([-np.pi, np.pi], 0, trigger_value/1000., facecolor='gray', edgecolor='none', alpha=0.3)

        ax.set_rasterization_zorder(ZORDER)
        fifi.mpl_functions.set_fontsize(ax, 6)

        layout.append_figure_to_layer(layout.figures['speed_vs_time'], 'speed_vs_time', cleartarget=True)

        ###################################################################################################
        # altitude
        ax = layout.axes[('altitude_vs_time', 'altitude_vs_time')]
        braid_analysis_plots.plot_column_vs_time(df_3d_slice,
                                                    column='z',
                                                    time_key='time_relative_to_flash',
                                                    norm_columns_to_min_max=True,
                                                    cmap='bone_r',
                                                    vmin=0, vmax=1,
                                                    bin_y= np.arange(0, 0.6+0.01, 0.01),
                                                    bin_x= np.arange(-0.5+dt/2, tmax+dt+dt/2, dt),
                                                    ax=ax  )

        if trigger_value != 0:
            ax.fill_betweenx([-np.pi, np.pi], 0, trigger_value/1000., facecolor='red', edgecolor='none', alpha=0.3)
        else:
            ax.fill_betweenx([-np.pi, np.pi], 0, trigger_value/1000., facecolor='gray', edgecolor='none', alpha=0.3)

        ax.set_rasterization_zorder(ZORDER)
        fifi.mpl_functions.set_fontsize(ax, 6)

        layout.append_figure_to_layer(layout.figures['altitude_vs_time'], 'altitude_vs_time', cleartarget=True)

        ###################################################################################################
        # save plots
        layout.write_svg(output_figure_name)
        cairosvg.svg2pdf(url=output_figure_name, write_to=output_figure_name.split('.svg')[0]+'.pdf')
