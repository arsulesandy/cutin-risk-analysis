"""Interactive highD playback entrypoint with optional SFC overlay support."""

import os
import sys
import pickle
import argparse

from read_csv import *
from visualize_frame import VisualizationPlot
from cutin_risk.paths import (
    dataset_root_path,
    highd_background_image,
    highd_pickle_path,
    highd_recording_meta_csv,
    highd_tracks_csv,
    highd_tracks_meta_csv,
    step14_codes_csv_path,
)


def _normalize_recording_id(recording_id: str) -> str:
    """Normalize a recording identifier to canonical two-digit highD form."""
    rid = str(recording_id).strip()
    if not rid:
        raise ValueError("Recording id must not be empty.")
    return f"{int(rid):02d}" if rid.isdigit() else rid


def _discover_recording_ids() -> list[str]:
    """Discover available recording ids by scanning `*_tracks.csv` files."""
    root = dataset_root_path()
    if not root.exists():
        return []
    recording_ids = []
    for path in sorted(root.glob("*_tracks.csv")):
        prefix = path.name[:-len("_tracks.csv")]
        if prefix.isdigit():
            recording_ids.append(f"{int(prefix):02d}")
    return recording_ids


def _resolve_highd_recording_paths(arguments: dict, recording_id: str) -> dict:
    """Fill CLI argument dictionary with resolved paths for one recording."""
    rid = _normalize_recording_id(recording_id)
    resolved_arguments = dict(arguments)
    resolved_arguments["recording_id"] = rid
    resolved_arguments["input_path"] = str(highd_tracks_csv(rid))
    resolved_arguments["input_static_path"] = str(highd_tracks_meta_csv(rid))
    resolved_arguments["input_meta_path"] = str(highd_recording_meta_csv(rid))
    resolved_arguments["pickle_path"] = str(highd_pickle_path(rid))
    resolved_arguments["background_image"] = str(highd_background_image(rid))
    if not resolved_arguments.get("sfc_codes_csv"):
        resolved_arguments["sfc_codes_csv"] = str(step14_codes_csv_path())
    return resolved_arguments


def _load_tracks(arguments: dict):
    """Load tracks from pickle when available, otherwise parse CSV."""
    print("Try to find the saved pickle file for better performance.")
    if os.path.exists(arguments["pickle_path"]):
        with open(arguments["pickle_path"], "rb") as fp:
            tracks = pickle.load(fp)
        print("Found pickle file {}.".format(arguments["pickle_path"]))
    else:
        print("Pickle file not found, csv will be imported now.")
        tracks = read_track_csv(arguments)
        print("Finished importing the pickle file.")
        if arguments["save_as_pickle"]:
            print("Save tracks to pickle file.")
            with open(arguments["pickle_path"], "wb") as fp:
                pickle.dump(tracks, fp)
    return tracks


def _load_highd_recording(base_arguments: dict, recording_id: str):
    """Load tracks plus static/meta tables for one recording id."""
    loaded_arguments = _resolve_highd_recording_paths(base_arguments, recording_id)
    print("Loading highD recording {}.".format(loaded_arguments["recording_id"]))

    tracks = _load_tracks(loaded_arguments)

    try:
        static_info = read_static_info(loaded_arguments)
    except Exception as exc:
        raise RuntimeError("The static info file is either missing or contains incorrect characters.") from exc

    try:
        meta_dictionary = read_meta_info(loaded_arguments)
    except Exception as exc:
        raise RuntimeError("The video meta file is either missing or contains incorrect characters.") from exc

    return loaded_arguments, tracks, static_info, meta_dictionary


def create_args():
    """Build and resolve CLI arguments for the visualizer application."""
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
    base_arguments = dict(created_arguments)

    try:
        tracks = _load_tracks(created_arguments)
    except Exception:
        print("Failed to read tracks from csv/pickle input.")
        sys.exit(1)

    try:
        static_info = read_static_info(created_arguments)
    except Exception:
        print("The static info file is either missing or contains incorrect characters.")
        sys.exit(1)

    try:
        meta_dictionary = read_meta_info(created_arguments)
    except Exception:
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
        recording_options = _discover_recording_ids()
        visualization_plot = VisualizationPlot(
            created_arguments,
            tracks,
            static_info,
            meta_dictionary,
            recording_loader=lambda rid: _load_highd_recording(base_arguments, rid),
            recording_options=recording_options,
        )
        visualization_plot.show()
