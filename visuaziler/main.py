import os
import sys
import pickle
import argparse

from read_csv import *
from visualize_frame import VisualizationPlot
from cutin_risk.paths import (
    highd_background_image,
    highd_pickle_path,
    highd_recording_meta_csv,
    highd_tracks_csv,
    highd_tracks_meta_csv,
    step14_codes_csv_path,
)


def create_args():
    parser = argparse.ArgumentParser(description="ParameterOptimizer")
    parser.add_argument(
        '--recording_id',
        default="01",
        type=str,
        help='highD recording id used for default file paths (e.g., 01).'
    )
    # --- Input paths ---
    parser.add_argument('--input_path', default=None, type=str,
                        help='CSV file of the tracks')
    parser.add_argument('--input_static_path', default=None,
                        type=str,
                        help='Static meta data file for each track')
    parser.add_argument('--input_meta_path', default=None,
                        type=str,
                        help='Static meta data file for the whole video')
    parser.add_argument('--pickle_path', default=None, type=str,
                        help='Converted pickle file that contains corresponding information of the "input_path" file')
    parser.add_argument('--sfc_codes_csv',
                        default=None,
                        type=str,
                        help='Optional CSV containing per-frame SFC binary codes used to print decoded 3x3 matrices.')
    # --- Settings ---
    parser.add_argument('--visualize', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='True if you want to visualize the data.')
    parser.add_argument('--background_image', default=None, type=str,
                        help='Optional: you can specify the correlating background image.')

    # --- Visualization settings ---
    parser.add_argument('--plotBoundingBoxes', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the bounding boxes or not.')
    parser.add_argument('--plotDirectionTriangle', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the direction triangle or not.')
    parser.add_argument('--plotTextAnnotation', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the text annotation or not.')
    parser.add_argument('--plotDetailedLabel', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: enable detailed label (class/velocity/ID/lane). Default is short label ID/lane.')
    parser.add_argument('--plotIdOnlyLabel', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: draw ID on the vehicle box and keep lane label above the vehicle.')
    parser.add_argument('--plotTrackingLines', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: show/hide red tracking lines used for direction/lane-change trace.')
    parser.add_argument('--plotClass', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to show the class in the text annotation.')
    parser.add_argument('--plotVelocity', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to show the class in the text annotation.')
    parser.add_argument('--plotIDs', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to show the class in the text annotation.')

    # --- I/O settings ---
    parser.add_argument('--save_as_pickle', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: you can save the tracks as pickle.')
    parsed_arguments = vars(parser.parse_args())

    recording_id = str(parsed_arguments.get("recording_id", "01"))
    if not parsed_arguments.get("input_path"):
        parsed_arguments["input_path"] = str(highd_tracks_csv(recording_id))
    if not parsed_arguments.get("input_static_path"):
        parsed_arguments["input_static_path"] = str(highd_tracks_meta_csv(recording_id))
    if not parsed_arguments.get("input_meta_path"):
        parsed_arguments["input_meta_path"] = str(highd_recording_meta_csv(recording_id))
    if not parsed_arguments.get("pickle_path"):
        parsed_arguments["pickle_path"] = str(highd_pickle_path(recording_id))
    if not parsed_arguments.get("sfc_codes_csv"):
        parsed_arguments["sfc_codes_csv"] = str(step14_codes_csv_path())
    if not parsed_arguments.get("background_image"):
        parsed_arguments["background_image"] = str(highd_background_image(recording_id))

    return parsed_arguments


if __name__ == '__main__':
    created_arguments = create_args()
    print("Try to find the saved pickle file for better performance.")
    # Read the track csv and convert to useful format
    if os.path.exists(created_arguments["pickle_path"]):
        with open(created_arguments["pickle_path"], "rb") as fp:
            tracks = pickle.load(fp)
        print("Found pickle file {}.".format(created_arguments["pickle_path"]))
    else:
        print("Pickle file not found, csv will be imported now.")
        tracks = read_track_csv(created_arguments)
        print("Finished importing the pickle file.")

    if created_arguments["save_as_pickle"] and not os.path.exists(created_arguments["pickle_path"]):
        print("Save tracks to pickle file.")
        with open(created_arguments["pickle_path"], "wb") as fp:
            pickle.dump(tracks, fp)

    # Read the static info
    try:
        static_info = read_static_info(created_arguments)
    except:
        print("The static info file is either missing or contains incorrect characters.")
        sys.exit(1)

    # Read the video meta
    try:
        meta_dictionary = read_meta_info(created_arguments)
    except:
        print("The video meta file is either missing or contains incorrect characters.")
        sys.exit(1)

    if created_arguments["visualize"]:
        if tracks is None:
            print("Please specify the path to the tracks csv/pickle file.")
            sys.exit(1)
        if static_info is None:
            print("Please specify the path to the static tracks csv file.")
            sys.exit(1)
        if meta_dictionary is None:
            print("Please specify the path to the video meta csv file.")
            sys.exit(1)
        visualization_plot = VisualizationPlot(created_arguments, tracks, static_info, meta_dictionary)
        visualization_plot.show()
