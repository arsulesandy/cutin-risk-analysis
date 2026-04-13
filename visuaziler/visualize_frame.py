"""Matplotlib-based interactive frame viewer for highD and exiD tracks."""

import os
import re
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.pyplot import imread
from matplotlib.widgets import Button, Slider

from read_csv import *

mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.titleweight"] = "bold"


class VisualizationPlot(object):
    """Interactive UI for stepping through frames and inspecting vehicle context."""

    HILBERT_BIT_LAYOUT = (
        (0, 1, 14, 15),
        (3, 2, 13, 12),
        (4, 7, 8, 11),
        (5, 6, 9, 10),
    )

    THEME = {
        "figure_bg": "#060b12",
        "road_bg": "#0a1018",
        "panel": "#0f1a27",
        "panel_soft": "#152435",
        "panel_edge": "#2a3f57",
        "text_main": "#e2e8f0",
        "text_muted": "#94a3b8",
        "text_dim": "#64748b",
        "button": "#1b2b3d",
        "button_hover": "#2a3f56",
        "button_active": "#0f1724",
        "button_text": "#f1f5f9",
        "accent": "#14b8a6",
        "accent_alt": "#38bdf8",
        "danger": "#f43f5e",
        "success": "#22c55e",
        "cutter": "#f59e0b",
        "cutter_trail": "#60a5fa",
        "neighbor": "#38bdf8",
        "risk_critical": "#ef4444",
        "risk_elevated": "#f59e0b",
        "risk_normal": "#22c55e",
    }

    def __init__(
        self,
        arguments,
        read_tracks,
        static_info,
        meta_dictionary,
        recording_loader=None,
        recording_options=None,
        dataset_loader=None,
        dataset_recording_options=None,
        fig=None,
    ):
        self.arguments = arguments
        self.tracks = read_tracks
        self.static_info = static_info
        self.meta_dictionary = meta_dictionary
        self.recording_loader = recording_loader
        self.dataset_loader = dataset_loader
        self.active_dataset_name = self._normalize_dataset_name(
            arguments.get("dataset") or meta_dictionary.get(DATASET_NAME, "highd")
        )
        self.selected_dataset_name = self.active_dataset_name
        self.dataset_recording_options = self._normalize_dataset_recording_options(dataset_recording_options)
        if recording_options and self.active_dataset_name not in self.dataset_recording_options:
            self.dataset_recording_options[self.active_dataset_name] = self._normalize_recording_options(recording_options)
        self.recording_options = self._recording_options_for_dataset(self.selected_dataset_name)
        self.active_recording_id = self._resolve_recording_id(arguments)
        if self.active_recording_id and self.active_recording_id not in self.recording_options:
            self.recording_options.append(self.active_recording_id)
            self.recording_options = sorted(self.recording_options)
        self.selected_recording_id = self.active_recording_id
        self.track_lookup = self._build_track_lookup(self.tracks)
        self.maximum_frames = self._compute_maximum_frame()
        self.current_frame = 1
        self.changed_button = False
        self.plotted_objects = []
        self.pick_event_cid = None
        self.key_event_cid = None
        self.mouse_event_cid = None
        self.scroll_event_cid = None
        self.playing = False
        self.playback_fps = 12
        self.playback_timer = None
        self.sfc_eval_frame = None
        self.sfc_eval_index = 0
        self.sfc_eval_count = 0
        self.default_xlim = None
        self.default_ylim = None
        self.timeline_event_points = []
        self.timeline_seek_line = None
        self.timeline_stage_colors = {
            "decision": "#38bdf8",
            "execution": "#f59e0b",
            "merge": "#14b8a6",
            "other": "#64748b",
        }
        self.sfc_canonical_user = False
        self.sfc_canonical_effective = False
        self.sfc_orientation_inferred = None
        self.sfc_orientation_note = None
        self.sfc_codes_by_frame = self._load_sfc_codes_by_frame(
            arguments.get("sfc_codes_csv"),
            arguments.get("input_path"),
        )
        self._resolve_effective_sfc_mode()
        self.timeline_event_points = self._build_timeline_event_points()
        self.semantic_stats = {
            "visible": 0,
            "cutters": 0,
            "neighbors": 0,
            "critical": 0,
            "elevated": 0,
            "normal": 0,
        }

        # Allocate more vertical room to the SFC/status information band.
        # This reduces text overlap on smaller laptop screens.
        self.road_bottom = 0.475
        self.road_top = 0.885
        self.header_bottom = 0.89
        self.header_height = 0.075
        self.info_bottom = 0.205
        self.info_height = 0.205
        self.timeline_bottom = 0.122
        self.timeline_height = 0.03
        self.controls_bottom = 0.055
        self.controls_height = 0.065

        if fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self.fig.set_size_inches(24, 9)
        else:
            self.fig = fig
            self.ax = self.fig.gca()
        try:
            self.fig.canvas.manager.set_window_title("{} Console".format(self._dataset_display_name()))
        except Exception:
            pass
        self._center_window_on_screen()

        self.fig.patch.set_facecolor(self.THEME["figure_bg"])
        self.ax.set_position([0.0, self.road_bottom, 1.0, self.road_top - self.road_bottom])
        self.ax.set_facecolor(self.THEME["road_bg"])

        self._build_header()
        self._build_timeline()
        self._build_info_panels()
        self._draw_background()
        self._build_controls()
        self._build_playback_timer()
        self._bind_callbacks()

        self.update_figure()
        self.ax.set_autoscale_on(False)

    @staticmethod
    def _normalize_recording_id(recording_id):
        rid = str(recording_id).strip()
        if not rid:
            return None
        if rid.isdigit():
            return "{:02d}".format(int(rid))
        return rid

    @staticmethod
    def _normalize_dataset_name(dataset_name):
        token = str(dataset_name or "highd").strip().lower()
        if token not in {"highd", "exid"}:
            return "highd"
        return token

    def _normalize_dataset_recording_options(self, dataset_recording_options):
        if not dataset_recording_options:
            return {}
        normalized = {}
        for dataset_name, recording_options in dataset_recording_options.items():
            normalized[self._normalize_dataset_name(dataset_name)] = self._normalize_recording_options(recording_options)
        return normalized

    def _recording_options_for_dataset(self, dataset_name):
        return list(self.dataset_recording_options.get(self._normalize_dataset_name(dataset_name), []))

    def _dataset_display_name(self, dataset_name=None):
        dataset = self._normalize_dataset_name(dataset_name or self.active_dataset_name)
        return "exiD" if dataset == "exid" else "highD"

    def _normalize_recording_options(self, recording_options):
        if not recording_options:
            return []
        normalized = []
        for rid in recording_options:
            norm_rid = self._normalize_recording_id(rid)
            if norm_rid is not None:
                normalized.append(norm_rid)
        return sorted(set(normalized))

    def _resolve_recording_id(self, arguments):
        from_path = self._infer_recording_id(arguments.get("input_path"))
        if from_path is not None:
            return "{:02d}".format(int(from_path))
        return self._normalize_recording_id(arguments.get("recording_id"))

    @staticmethod
    def _track_id_as_int(track):
        raw_track_id = track[TRACK_ID]
        if isinstance(raw_track_id, (list, tuple)):
            return int(raw_track_id[0])
        if hasattr(raw_track_id, "__len__") and not isinstance(raw_track_id, (str, bytes)):
            return int(raw_track_id[0])
        return int(raw_track_id)

    def _build_track_lookup(self, tracks):
        lookup = {}
        for track in tracks or []:
            try:
                lookup[self._track_id_as_int(track)] = track
            except Exception:
                continue
        return lookup

    def _compute_maximum_frame(self):
        if self.static_info is None or self.tracks is None:
            return 1
        max_frame = 1
        for track in self.tracks:
            try:
                track_id = self._track_id_as_int(track)
                final_frame = int(self.static_info[track_id][FINAL_FRAME]) - 1
                if final_frame > max_frame:
                    max_frame = final_frame
            except Exception:
                continue
        return max(1, max_frame)

    def _build_header(self):
        self.ax_header = self.fig.add_axes([0.02, self.header_bottom, 0.96, self.header_height])
        self.ax_header.set_facecolor(self.THEME["panel"])
        self.ax_header.set_xticks([])
        self.ax_header.set_yticks([])
        for spine in self.ax_header.spines.values():
            spine.set_color(self.THEME["panel_edge"])
            spine.set_linewidth(1.0)

        self.header_title_text = self.ax_header.text(
            0.015,
            0.62,
            "",
            transform=self.ax_header.transAxes,
            ha="left",
            va="center",
            fontsize=16,
            color=self.THEME["text_main"],
            fontweight="bold",
        )
        self.header_controls_text = self.ax_header.text(
            0.015,
            0.22,
            "Controls: Space play/pause | Left/Right +/-1 | Up/Down +/-10 | L load recording",
            transform=self.ax_header.transAxes,
            ha="left",
            va="center",
            fontsize=9.2,
            color=self.THEME["text_muted"],
        )
        self.header_note_text = self.ax_header.text(
            0.62,
            0.22,
            "",
            transform=self.ax_header.transAxes,
            ha="left",
            va="center",
            fontsize=8.8,
            color=self.THEME["accent_alt"],
        )
        self.header_state_text = self.ax_header.text(
            0.985,
            0.62,
            "",
            transform=self.ax_header.transAxes,
            ha="right",
            va="center",
            fontsize=10.5,
            color=self.THEME["text_main"],
            fontweight="bold",
        )

        self.ax_button_dataset_highd = self.fig.add_axes([0.79, self.header_bottom + 0.022, 0.07, 0.032])
        self.ax_button_dataset_exid = self.fig.add_axes([0.865, self.header_bottom + 0.022, 0.07, 0.032])
        self.button_dataset_highd = Button(self.ax_button_dataset_highd, "highD")
        self.button_dataset_exid = Button(self.ax_button_dataset_exid, "exiD")
        self._style_button(self.button_dataset_highd)
        self._style_button(self.button_dataset_exid)
        self._refresh_dataset_buttons()
        self._update_header_copy()

    def _center_window_on_screen(self):
        """
        Attempt to center the Matplotlib GUI window on screen.
        Supports common backends (Tk, Qt, Wx) and safely no-ops otherwise.
        """
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is None:
            return
        window = getattr(manager, "window", None)
        if window is None:
            return

        # Tk backend
        try:
            if hasattr(window, "wm_geometry") and hasattr(window, "winfo_screenwidth"):
                window.update_idletasks()
                width = int(window.winfo_width())
                height = int(window.winfo_height())
                if width <= 1 or height <= 1:
                    width = int(window.winfo_reqwidth())
                    height = int(window.winfo_reqheight())
                screen_w = int(window.winfo_screenwidth())
                screen_h = int(window.winfo_screenheight())
                x = max(0, (screen_w - width) // 2)
                y = max(0, (screen_h - height) // 2)
                window.wm_geometry(f"+{x}+{y}")
                return
        except Exception:
            pass

    def _header_note(self):
        if self._normalize_dataset_name(self.active_dataset_name) == "exid":
            return str(
                self.meta_dictionary.get(
                    ROAD_INFO_NOTE,
                    "exiD exploratory adapter | location-specific background crop active",
                )
            )
        return "Lane Directions: Upper=Dir1 (x<0) | Lower=Dir2 (x>0)"

    def _update_header_copy(self):
        if hasattr(self, "header_title_text"):
            self.header_title_text.set_text("{} Scenario Intelligence Console".format(self._dataset_display_name()))
        if hasattr(self, "header_note_text"):
            self.header_note_text.set_text(self._header_note())

    def _refresh_dataset_buttons(self):
        active = self._normalize_dataset_name(self.selected_dataset_name)
        button_specs = [
            (getattr(self, "button_dataset_highd", None), "highd"),
            (getattr(self, "button_dataset_exid", None), "exid"),
        ]
        for button, dataset_name in button_specs:
            if button is None:
                continue
            is_active = dataset_name == active
            face = self.THEME["accent"] if is_active else self.THEME["button"]
            button.ax.set_facecolor(face)
            button.color = face
            button.hovercolor = self.THEME["button_hover"]
            button.label.set_color(self.THEME["button_text"])

    def _switch_selected_dataset(self, dataset_name):
        target = self._normalize_dataset_name(dataset_name)
        if target == self.selected_dataset_name:
            return
        self.selected_dataset_name = target
        self.recording_options = self._recording_options_for_dataset(target)
        if self.recording_options:
            if self.selected_recording_id not in self.recording_options:
                self.selected_recording_id = self.recording_options[0]
        else:
            self.selected_recording_id = None
        self.recording_slider = self._build_recording_slider()
        if self.recording_slider is not None:
            self.recording_slider.on_changed(self.update_recording_slider)
        self._refresh_dataset_buttons()
        self._refresh_control_value_labels()
        if hasattr(self, "status_text"):
            self.status_text.set_text(self._build_status_text())
        self.fig.canvas.draw_idle()

        # Qt backend
        try:
            if hasattr(window, "move") and hasattr(window, "width") and hasattr(window, "height"):
                from matplotlib.backends.qt_compat import QtWidgets

                app = QtWidgets.QApplication.instance()
                screen = app.primaryScreen() if app is not None else None
                if screen is not None:
                    rect = screen.availableGeometry()
                    width = int(window.width())
                    height = int(window.height())
                    x = int(rect.x() + (rect.width() - width) / 2)
                    y = int(rect.y() + (rect.height() - height) / 2)
                    window.move(x, y)
                    return
        except Exception:
            pass

        # Wx backend
        try:
            if hasattr(window, "GetDisplaySize") and hasattr(window, "GetSize") and hasattr(window, "SetPosition"):
                screen_w, screen_h = window.GetDisplaySize()
                width, height = window.GetSize()
                x = max(0, int((screen_w - width) / 2))
                y = max(0, int((screen_h - height) / 2))
                window.SetPosition((x, y))
                return
        except Exception:
            pass

    def _build_timeline(self):
        timeline_y = self.info_bottom + self.info_height + 0.008
        self.ax_timeline = self.fig.add_axes([0.02, timeline_y, 0.96, 0.016])
        self.ax_timeline.set_facecolor(self.THEME["panel_soft"])
        self.ax_timeline.set_xlim(1, max(2, int(self.maximum_frames)))
        self.ax_timeline.set_ylim(0, 1)
        self.ax_timeline.set_xticks([])
        self.ax_timeline.set_yticks([])
        for spine in self.ax_timeline.spines.values():
            spine.set_color(self.THEME["panel_edge"])
            spine.set_linewidth(0.8)
        self._refresh_timeline_plot()

    def _build_info_panels(self):
        self.ax_sfc_info = self.fig.add_axes([0.02, self.info_bottom, 0.72, self.info_height])
        self.ax_status = self.fig.add_axes([0.75, self.info_bottom, 0.23, self.info_height])

        for panel in (self.ax_sfc_info, self.ax_status):
            panel.set_facecolor(self.THEME["panel_soft"])
            panel.set_xticks([])
            panel.set_yticks([])
            for spine in panel.spines.values():
                spine.set_color(self.THEME["panel_edge"])
                spine.set_linewidth(1.0)

        self.sfc_info_text = self.ax_sfc_info.text(
            0.02,
            0.97,
            "",
            transform=self.ax_sfc_info.transAxes,
            ha="left",
            va="top",
            fontsize=9.1,
            fontfamily="monospace",
            color=self.THEME["text_main"],
        )

        self.status_text = self.ax_status.text(
            0.03,
            0.95,
            "",
            transform=self.ax_status.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            color=self.THEME["text_main"],
            linespacing=1.25,
        )
        self.sfc_match_text = self.ax_status.text(
            0.03,
            0.26,
            "",
            transform=self.ax_status.transAxes,
            ha="left",
            va="top",
            fontsize=10.2,
            color=self.THEME["text_muted"],
            fontweight="bold",
        )
        self._draw_status_legend()

    def _build_controls(self):
        control_gap = 0.01
        button_height = 0.048

        self.ax_recording_slider = self.fig.add_axes(
            [0.02, self.timeline_bottom, 0.145, self.timeline_height], facecolor=self.THEME["panel_soft"]
        )
        self.ax_button_load_recording = self.fig.add_axes([0.17, self.timeline_bottom - 0.004, 0.11, 0.038])
        self.ax_frame_slider = self.fig.add_axes(
            [0.30, self.timeline_bottom, 0.68, self.timeline_height], facecolor=self.THEME["panel_soft"]
        )

        prev10_x = 0.30
        prev_x = prev10_x + 0.075 + control_gap
        play_x = prev_x + 0.055 + control_gap
        next_x = play_x + 0.07 + control_gap
        next10_x = next_x + 0.055 + control_gap

        self.ax_button_previous2 = self.fig.add_axes([prev10_x, self.controls_bottom, 0.075, button_height])
        self.ax_button_previous = self.fig.add_axes([prev_x, self.controls_bottom, 0.055, button_height])
        self.ax_button_play_pause = self.fig.add_axes([play_x, self.controls_bottom, 0.07, button_height])
        self.ax_button_next = self.fig.add_axes([next_x, self.controls_bottom, 0.055, button_height])
        self.ax_button_next2 = self.fig.add_axes([next10_x, self.controls_bottom, 0.075, button_height])
        self.ax_speed_slider = self.fig.add_axes(
            [0.73, self.controls_bottom + 0.01, 0.25, 0.03], facecolor=self.THEME["panel_soft"]
        )

        self.frame_slider = self._build_frame_slider()
        self.recording_slider = self._build_recording_slider()
        self.speed_slider = self._build_speed_slider()

        self.button_previous2 = Button(self.ax_button_previous2, "<< 10")
        self.button_previous = Button(self.ax_button_previous, "<")
        self.button_play_pause = Button(self.ax_button_play_pause, "Play")
        self.button_next = Button(self.ax_button_next, ">")
        self.button_next2 = Button(self.ax_button_next2, "10 >>")
        self.button_load_recording = Button(self.ax_button_load_recording, "Load recording")

        for button in (
            self.button_previous2,
            self.button_previous,
            self.button_play_pause,
            self.button_next,
            self.button_next2,
            self.button_load_recording,
        ):
            self._style_button(button)
        self._refresh_control_value_labels()

    def _draw_status_legend(self):
        legend_specs = [
            ("CUTTER", self.THEME["cutter"]),
            ("TRAFFIC", self.THEME["danger"]),
        ]
        x = 0.03
        y = 0.03
        swatch_w = 0.018
        swatch_h = 0.075
        stride = 0.24
        for label, color in legend_specs:
            swatch = patches.Rectangle(
                (x, y),
                swatch_w,
                swatch_h,
                transform=self.ax_status.transAxes,
                facecolor=color,
                edgecolor=self.THEME["panel_edge"],
                lw=0.6,
                alpha=0.95,
                zorder=2,
                clip_on=False,
            )
            self.ax_status.add_patch(swatch)
            self.ax_status.text(
                x + swatch_w + 0.01,
                y + (swatch_h / 2),
                label,
                transform=self.ax_status.transAxes,
                ha="left",
                va="center",
                fontsize=7.6,
                color=self.THEME["text_muted"],
                zorder=3,
            )
            x += stride

    @staticmethod
    def _stage_key(stage_value):
        if stage_value is None:
            return "other"
        stage_text = str(stage_value).strip().lower()
        if "decision" in stage_text:
            return "decision"
        if "execution" in stage_text:
            return "execution"
        if "merge" in stage_text or "lane" in stage_text:
            return "merge"
        return "other"

    def _responsive_scale(self):
        """Return a conservative UI scale factor based on current figure size."""
        try:
            width_in, height_in = self.fig.get_size_inches()
        except Exception:
            return 1.0
        width_scale = float(width_in) / 24.0
        height_scale = float(height_in) / 9.0
        return float(max(0.72, min(1.0, min(width_scale, height_scale))))

    def _responsive_font(self, base_size):
        """Scale font sizes down on smaller windows to avoid overlap."""
        return max(6.8, float(base_size) * self._responsive_scale())

    @staticmethod
    def _ellipsize(text, max_chars):
        if text is None:
            return ""
        raw = str(text)
        if len(raw) <= int(max_chars):
            return raw
        return raw[: max(0, int(max_chars) - 3)] + "..."

    @staticmethod
    def _coerce_bool(value):
        """Parse bool-like config values robustly (bool, numeric, or common strings)."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        token = str(value).strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off", ""}:
            return False
        return bool(value)

    def _infer_sfc_orientation(self, sample_limit=320):
        """
        Infer whether loaded SFC codes are raw Step14 orientation or canonical Step15A orientation.

        Returns a tuple: (inferred_mode, sample_count, match_raw, match_canonical)
        where inferred_mode is one of {"raw", "canonical"}.
        """
        if self._normalize_dataset_name(self.active_dataset_name) != "highd":
            return None
        if not self.sfc_codes_by_frame:
            return None

        sample_count = 0
        match_raw = 0
        match_canonical = 0
        for frame in sorted(int(k) for k in self.sfc_codes_by_frame.keys()):
            entries = self.sfc_codes_by_frame.get(frame, [])
            for entry in entries:
                if sample_count >= int(sample_limit):
                    break
                cutter_id = entry.get("cutter_id")
                if cutter_id is None:
                    continue

                reference_matrix = self._build_highd_reference_matrix(cutter_id, int(frame))
                if reference_matrix is None:
                    continue

                try:
                    raw_matrix = np.asarray(self._decode_code_to_3x3_matrix(entry["code"]), dtype=int)
                except Exception:
                    continue
                rtl = self._is_track_right_to_left(cutter_id, int(frame))
                raw_mode_matrix = np.fliplr(raw_matrix) if rtl else raw_matrix

                if np.array_equal(raw_mode_matrix, np.asarray(reference_matrix, dtype=int)):
                    match_raw += 1
                if np.array_equal(raw_matrix, np.asarray(reference_matrix, dtype=int)):
                    match_canonical += 1
                sample_count += 1
            if sample_count >= int(sample_limit):
                break

        if sample_count <= 0:
            return None
        inferred_mode = "raw" if match_raw >= match_canonical else "canonical"
        return inferred_mode, int(sample_count), int(match_raw), int(match_canonical)

    def _resolve_effective_sfc_mode(self):
        """Resolve effective SFC code mode and auto-correct obvious flag/code-table mismatches."""
        self.sfc_canonical_user = self._coerce_bool(self.arguments.get("sfc_codes_canonical", False))
        self.sfc_canonical_effective = bool(self.sfc_canonical_user)
        self.sfc_orientation_inferred = None
        self.sfc_orientation_note = None

        inferred = self._infer_sfc_orientation(sample_limit=320)
        if inferred is None:
            return
        inferred_mode, sample_count, match_raw, match_canonical = inferred
        self.sfc_orientation_inferred = inferred_mode

        preferred_canonical = bool(inferred_mode == "canonical")
        preferred_matches = match_canonical if preferred_canonical else match_raw
        other_matches = match_raw if preferred_canonical else match_canonical
        margin = int(preferred_matches - other_matches)
        # Auto-correct only when evidence is strong enough.
        strong_margin = max(12, int(round(0.18 * sample_count)))
        if (sample_count >= 40) and (margin >= strong_margin) and (self.sfc_canonical_user != preferred_canonical):
            self.sfc_canonical_effective = preferred_canonical
            self.sfc_orientation_note = (
                "SFC mode auto-corrected (flag={}, inferred={} from {} samples).".format(
                    "canonical" if self.sfc_canonical_user else "raw",
                    inferred_mode,
                    sample_count,
                )
            )

    def _build_timeline_event_points(self):
        if not self.sfc_codes_by_frame:
            return []

        frames = sorted(int(frame) for frame in self.sfc_codes_by_frame.keys())
        max_markers = 300
        stride = max(1, int(np.ceil(len(frames) / max_markers)))
        sampled_frames = frames[::stride]
        if sampled_frames[-1] != frames[-1]:
            sampled_frames.append(frames[-1])

        event_points = []
        for frame in sampled_frames:
            entries = self.sfc_codes_by_frame.get(frame, [])
            stage_keys = {self._stage_key(entry.get("stage")) for entry in entries}
            stage = "other"
            if "decision" in stage_keys:
                stage = "decision"
            elif "execution" in stage_keys:
                stage = "execution"
            elif "merge" in stage_keys:
                stage = "merge"
            event_points.append((int(frame), stage))
        return event_points

    def _refresh_timeline_plot(self):
        self.ax_timeline.cla()
        self.ax_timeline.set_facecolor(self.THEME["panel_soft"])
        self.ax_timeline.set_xlim(1, max(2, int(self.maximum_frames)))
        self.ax_timeline.set_ylim(0, 1)
        self.ax_timeline.set_xticks([])
        self.ax_timeline.set_yticks([])
        for spine in self.ax_timeline.spines.values():
            spine.set_color(self.THEME["panel_edge"])
            spine.set_linewidth(0.8)

        self.ax_timeline.plot(
            [1, max(2, int(self.maximum_frames))],
            [0.5, 0.5],
            color=self.THEME["panel_edge"],
            linewidth=1.1,
            alpha=0.9,
            zorder=1,
        )

        for frame, stage in self.timeline_event_points:
            color = self.timeline_stage_colors.get(stage, self.timeline_stage_colors["other"])
            self.ax_timeline.vlines(
                frame,
                0.15,
                0.85,
                color=color,
                linewidth=1.05,
                alpha=0.9,
                zorder=2,
            )

        self.timeline_seek_line = self.ax_timeline.axvline(
            int(self.current_frame),
            color=self.THEME["accent"],
            linewidth=2.1,
            alpha=0.95,
            zorder=3,
        )

    def _update_timeline_cursor(self):
        if self.timeline_seek_line is None:
            return
        x = int(self.current_frame)
        self.timeline_seek_line.set_xdata([x, x])

    def _build_playback_timer(self):
        self.playback_timer = self.fig.canvas.new_timer(interval=self._fps_to_interval(self.playback_fps))
        self.playback_timer.add_callback(self._on_playback_tick)

    def _bind_callbacks(self):
        self.frame_slider.on_changed(self.update_slider)
        if self.recording_slider is not None:
            self.recording_slider.on_changed(self.update_recording_slider)
        self.speed_slider.on_changed(self.update_speed_slider)

        self.button_previous.on_clicked(self.update_button_previous)
        self.button_next.on_clicked(self.update_button_next)
        self.button_previous2.on_clicked(self.update_button_previous2)
        self.button_next2.on_clicked(self.update_button_next2)
        self.button_play_pause.on_clicked(self.update_button_play_pause)
        self.button_load_recording.on_clicked(self.update_button_load_recording)
        if hasattr(self, "button_dataset_highd"):
            self.button_dataset_highd.on_clicked(lambda _: self._switch_selected_dataset("highd"))
        if hasattr(self, "button_dataset_exid"):
            self.button_dataset_exid.on_clicked(lambda _: self._switch_selected_dataset("exid"))

        if self.key_event_cid is not None:
            self.fig.canvas.mpl_disconnect(self.key_event_cid)
        self.key_event_cid = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        if self.mouse_event_cid is not None:
            self.fig.canvas.mpl_disconnect(self.mouse_event_cid)
        self.mouse_event_cid = self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        if self.scroll_event_cid is not None:
            self.fig.canvas.mpl_disconnect(self.scroll_event_cid)
        self.scroll_event_cid = self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

    def _style_button(self, button):
        button.ax.set_facecolor(self.THEME["button"])
        button.color = self.THEME["button"]
        button.hovercolor = self.THEME["button_hover"]
        button.label.set_color(self.THEME["button_text"])
        button.label.set_fontsize(9.5)
        button.label.set_fontweight("bold")
        for spine in button.ax.spines.values():
            spine.set_color(self.THEME["panel_edge"])
            spine.set_linewidth(1.0)

    def _style_slider(self, slider):
        slider.poly.set_facecolor(self.THEME["accent"])
        slider.poly.set_alpha(0.95)
        slider.vline.set_color(self.THEME["accent_alt"])
        slider.vline.set_linewidth(1.8)
        slider.label.set_visible(False)
        slider.valtext.set_color(self.THEME["text_main"])
        slider.valtext.set_transform(slider.ax.transAxes)
        slider.valtext.set_position((0.988, 0.5))
        slider.valtext.set_ha("right")
        slider.valtext.set_va("center")
        slider.valtext.set_fontsize(9.3)
        slider.valtext.set_bbox(
            dict(boxstyle="round,pad=0.16", fc=self.THEME["panel"], ec=self.THEME["panel_edge"], lw=0.7, alpha=0.92)
        )
        if hasattr(slider, "track") and slider.track is not None:
            slider.track.set_color(self.THEME["text_dim"])
            slider.track.set_alpha(0.35)
        for spine in slider.ax.spines.values():
            spine.set_color(self.THEME["panel_edge"])
            spine.set_linewidth(0.9)

    def _decorate_slider_axis(self, ax, label):
        ax.text(
            0.0,
            1.42,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.4,
            color=self.THEME["text_muted"],
            fontweight="bold",
            clip_on=False,
        )

    def _set_sfc_eval_count(self, evaluations_count):
        self.sfc_eval_count = max(0, int(evaluations_count))
        if self.sfc_eval_count <= 0:
            self.sfc_eval_index = 0
        else:
            self.sfc_eval_index = int(max(0, min(self.sfc_eval_index, self.sfc_eval_count - 1)))

    def _step_sfc_eval(self, delta):
        if self.sfc_eval_count <= 1:
            return
        next_index = int(max(0, min(self.sfc_eval_index + int(delta), self.sfc_eval_count - 1)))
        if next_index == self.sfc_eval_index:
            return
        self.sfc_eval_index = next_index
        self._update_sfc_matrix_overlay()
        self.fig.canvas.draw_idle()

    def _build_frame_slider(self):
        self.ax_frame_slider.cla()
        self.ax_frame_slider.set_facecolor(self.THEME["panel_soft"])
        slider = Slider(
            self.ax_frame_slider,
            "Frame",
            1,
            max(1, int(self.maximum_frames)),
            valinit=max(1, int(self.current_frame)),
            valfmt="%0.0f",
            valstep=1,
        )
        self._style_slider(slider)
        self._decorate_slider_axis(self.ax_frame_slider, "Frame")
        return slider

    def _build_recording_slider(self):
        self.ax_recording_slider.cla()
        self.ax_recording_slider.set_facecolor(self.THEME["panel_soft"])
        if not self.recording_options:
            self.ax_recording_slider.set_axis_off()
            return None

        initial_recording = self.selected_recording_id
        if initial_recording not in self.recording_options:
            initial_recording = self.recording_options[0]
        if len(self.recording_options) == 1:
            self.ax_recording_slider.text(
                0.02,
                0.5,
                "Recording {}".format(initial_recording),
                transform=self.ax_recording_slider.transAxes,
                ha="left",
                va="center",
                fontsize=10,
                color=self.THEME["text_main"],
            )
            self.ax_recording_slider.set_axis_off()
            self.selected_recording_id = initial_recording
            return None

        initial_index = self.recording_options.index(initial_recording) + 1
        slider = Slider(
            self.ax_recording_slider,
            "Recording",
            1,
            len(self.recording_options),
            valinit=initial_index,
            valfmt="%0.0f",
            valstep=1,
        )
        self._style_slider(slider)
        self._decorate_slider_axis(self.ax_recording_slider, "Recording")
        self.selected_recording_id = initial_recording
        return slider

    def _build_speed_slider(self):
        self.ax_speed_slider.cla()
        self.ax_speed_slider.set_facecolor(self.THEME["panel_soft"])
        slider = Slider(
            self.ax_speed_slider,
            "Playback FPS",
            2,
            30,
            valinit=float(self.playback_fps),
            valfmt="%0.0f",
            valstep=1,
        )
        self._style_slider(slider)
        self._decorate_slider_axis(self.ax_speed_slider, "Playback FPS")
        return slider

    def _refresh_control_value_labels(self):
        if hasattr(self, "frame_slider") and self.frame_slider is not None:
            self.frame_slider.valtext.set_text("{}/{}".format(int(self.current_frame), int(self.maximum_frames)))
        if hasattr(self, "recording_slider") and self.recording_slider is not None:
            self.recording_slider.valtext.set_text("{}".format(self.selected_recording_id or "-"))
        if hasattr(self, "speed_slider") and self.speed_slider is not None:
            self.speed_slider.valtext.set_text("{} fps".format(int(self.playback_fps)))

    def _update_header_state(self):
        if not hasattr(self, "header_state_text"):
            return
        state = "PLAY" if self.playing else "PAUSE"
        self.header_state_text.set_text(
            "{} {} | FRAME {}/{} | {} @ {}fps".format(
                self._dataset_display_name(),
                self.active_recording_id or "-",
                int(self.current_frame),
                int(self.maximum_frames),
                state,
                int(self.playback_fps),
            )
        )

    def _draw_background(self):
        self.ax.clear()
        self.ax.set_position([0.0, self.road_bottom, 1.0, self.road_top - self.road_bottom])
        self.ax.set_facecolor(self.THEME["road_bg"])

        background_image_path = self.arguments.get("background_image")
        if background_image_path is not None and os.path.exists(background_image_path):
            self.background_image = imread(background_image_path)
            self.y_sign = 1
            self.ax.imshow(self.background_image[:, :, :], alpha=0.93)
            tint = np.zeros_like(self.background_image[:, :, :], dtype=float)
            tint[..., 0] = 0.04
            tint[..., 1] = 0.10
            tint[..., 2] = 0.18
            self.ax.imshow(tint, alpha=0.18)
        else:
            self.background_image = None
            self.y_sign = -1
            self.outer_line_thickness = 0.2
            self.lane_color = "#94a3b8"
            self.plot_highway()

        x_limits = self.meta_dictionary.get(BACKGROUND_X_LIMITS)
        y_limits = self.meta_dictionary.get(BACKGROUND_Y_LIMITS)
        if x_limits is not None and y_limits is not None:
            self.ax.set_xlim(x_limits)
            self.ax.set_ylim(y_limits)

        self.plot_highway_information()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_autoscale_on(False)
        self.default_xlim = tuple(self.ax.get_xlim())
        self.default_ylim = tuple(self.ax.get_ylim())

    def _background_scale(self):
        scale = self.meta_dictionary.get(BACKGROUND_SCALE)
        if scale is None:
            return 0.10106 * 4.0
        try:
            numeric = float(scale)
        except Exception:
            return 0.10106 * 4.0
        return numeric if numeric > 0 else 1.0

    def _display_bbox(self, bbox):
        arr = np.array(bbox, dtype=float)
        if self.background_image is None:
            return arr
        return arr / self._background_scale()

    def _reload_recording(self, loaded_arguments, loaded_tracks, loaded_static_info, loaded_meta_dictionary):
        self._toggle_playback(force_stop=True)
        self.remove_patches()
        self.arguments = loaded_arguments
        self.tracks = loaded_tracks
        self.static_info = loaded_static_info
        self.meta_dictionary = loaded_meta_dictionary
        self.track_lookup = self._build_track_lookup(self.tracks)
        self.active_dataset_name = self._normalize_dataset_name(
            loaded_arguments.get("dataset") or loaded_meta_dictionary.get(DATASET_NAME, self.active_dataset_name)
        )
        self.selected_dataset_name = self.active_dataset_name
        self.recording_options = self._recording_options_for_dataset(self.active_dataset_name)
        self.active_recording_id = self._resolve_recording_id(loaded_arguments)
        self.selected_recording_id = self.active_recording_id
        self.maximum_frames = self._compute_maximum_frame()
        self.current_frame = 1
        self.sfc_eval_frame = None
        self.sfc_eval_index = 0
        self.sfc_codes_by_frame = self._load_sfc_codes_by_frame(
            loaded_arguments.get("sfc_codes_csv"),
            loaded_arguments.get("input_path"),
        )
        self._resolve_effective_sfc_mode()
        self.timeline_event_points = self._build_timeline_event_points()
        self._refresh_timeline_plot()
        self._draw_background()
        self._refresh_dataset_buttons()
        self._update_header_copy()
        try:
            self.fig.canvas.manager.set_window_title("{} Console".format(self._dataset_display_name()))
        except Exception:
            pass

        self.frame_slider = self._build_frame_slider()
        self.frame_slider.on_changed(self.update_slider)
        self.recording_slider = self._build_recording_slider()
        if self.recording_slider is not None:
            self.recording_slider.on_changed(self.update_recording_slider)

        if self.recording_slider is not None and self.selected_recording_id in self.recording_options:
            selected_index = self.recording_options.index(self.selected_recording_id) + 1
            self.recording_slider.set_val(selected_index)
            self.recording_slider.valtext.set_text(self.selected_recording_id)

        self.update_figure()
        self._refresh_control_value_labels()
        self.fig.canvas.draw_idle()

    def _fps_to_interval(self, fps):
        fps = max(1, int(round(fps)))
        return int(max(25, round(1000.0 / fps)))

    def _apply_playback_interval(self, immediate=False):
        if self.playback_timer is None:
            return
        self.playback_timer.interval = self._fps_to_interval(self.playback_fps)
        # Some Matplotlib backends apply interval changes only on next start().
        if immediate and self.playing:
            self.playback_timer.stop()
            self.playback_timer.start()

    def _toggle_playback(self, force_stop=False):
        if force_stop:
            self.playing = False
            if self.playback_timer is not None:
                self.playback_timer.stop()
            self.button_play_pause.label.set_text("Play")
            if hasattr(self, "status_text"):
                self.status_text.set_text(self._build_status_text())
            self._update_header_state()
            self._refresh_control_value_labels()
            self.fig.canvas.draw_idle()
            return

        self.playing = not self.playing
        if self.playing:
            if self.playback_timer is not None:
                self._apply_playback_interval(immediate=False)
                self.playback_timer.start()
            self.button_play_pause.label.set_text("Pause")
        else:
            if self.playback_timer is not None:
                self.playback_timer.stop()
            self.button_play_pause.label.set_text("Play")
        if hasattr(self, "status_text"):
            self.status_text.set_text(self._build_status_text())
        self._update_header_state()
        self._refresh_control_value_labels()
        self.fig.canvas.draw_idle()

    def _on_playback_tick(self):
        if not self.playing:
            return
        if self.current_frame >= self.maximum_frames:
            self._toggle_playback(force_stop=True)
            return
        self.current_frame += 1
        self.changed_button = True
        self.trigger_update()

    def _track_center_for_frame(self, track_id, frame):
        track = self.track_lookup.get(int(track_id))
        if track is None:
            return None
        static_track_information = self.static_info.get(int(track_id))
        if static_track_information is None:
            return None

        initial_frame = int(static_track_information[INITIAL_FRAME])
        final_frame = int(static_track_information[FINAL_FRAME])
        if frame < initial_frame or frame >= final_frame:
            return None

        current_index = int(frame - initial_frame)
        if current_index < 0 or current_index >= len(track[BBOX]):
            return None

        bounding_box = self._display_bbox(track[BBOX][current_index])

        y_position = self.y_sign * bounding_box[1]
        vehicle_box_y = y_position + (self.y_sign * bounding_box[3] if self.y_sign < 0 else 0)
        center_x = bounding_box[0] + bounding_box[2] / 2
        center_y = vehicle_box_y + bounding_box[3] / 2
        return np.array([center_x, center_y], dtype=float)

    def _collect_active_cutter_centers(self, active_cutter_ids):
        centers = []
        for cutter_id in active_cutter_ids:
            center = self._track_center_for_frame(cutter_id, int(self.current_frame))
            if center is not None:
                centers.append(center)
        return centers

    @staticmethod
    def _clamp_view_window(center, span, limits):
        forward = float(limits[1]) >= float(limits[0])
        lim_min = min(float(limits[0]), float(limits[1]))
        lim_max = max(float(limits[0]), float(limits[1]))
        span = min(max(1e-6, float(span)), lim_max - lim_min)
        half = span / 2.0
        lo = float(center - half)
        hi = float(center + half)
        if lo < lim_min:
            hi += lim_min - lo
            lo = lim_min
        if hi > lim_max:
            lo -= hi - lim_max
            hi = lim_max
        lo = max(lim_min, lo)
        hi = min(lim_max, hi)
        if forward:
            return lo, hi
        return hi, lo

    @staticmethod
    def _finite_metric(series, index):
        if series is None:
            return None
        try:
            value = float(series[index])
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def _risk_class_for_track(self, track, index):
        ttc_value = self._finite_metric(track.get(TTC), index)
        thw_value = self._finite_metric(track.get(THW), index)

        if (ttc_value is not None and ttc_value <= 1.5) or (thw_value is not None and thw_value <= 0.6):
            return "critical"
        if (ttc_value is not None and ttc_value <= 3.0) or (thw_value is not None and thw_value <= 1.2):
            return "elevated"
        return "normal"

    def _collect_active_neighbor_ids(self, active_cutter_ids):
        neighbor_columns = (
            PRECEDING_ID,
            FOLLOWING_ID,
            LEFT_PRECEDING_ID,
            LEFT_ALONGSIDE_ID,
            LEFT_FOLLOWING_ID,
            RIGHT_PRECEDING_ID,
            RIGHT_ALONGSIDE_ID,
            RIGHT_FOLLOWING_ID,
        )
        frame = int(self.current_frame)
        neighbor_ids = set()
        for cutter_id in active_cutter_ids:
            try:
                cutter_id_int = int(cutter_id)
            except Exception:
                continue
            cutter_track = self.track_lookup.get(cutter_id_int)
            cutter_static = self.static_info.get(cutter_id_int)
            if cutter_track is None or cutter_static is None:
                continue
            initial_frame = int(cutter_static[INITIAL_FRAME])
            final_frame = int(cutter_static[FINAL_FRAME])
            if frame < initial_frame or frame >= final_frame:
                continue
            current_index = int(frame - initial_frame)
            for col in neighbor_columns:
                values = cutter_track.get(col)
                if values is None:
                    continue
                try:
                    raw = values[current_index]
                except Exception:
                    continue
                try:
                    neighbor_id = int(raw)
                except Exception:
                    try:
                        neighbor_id = int(float(raw))
                    except Exception:
                        continue
                if neighbor_id > 0 and neighbor_id != cutter_id_int:
                    neighbor_ids.add(neighbor_id)
        return neighbor_ids

    def _build_status_text(self):
        state = "PLAYING" if self.playing else "PAUSED"
        stats = self.semantic_stats or {}
        eval_summary = self._build_sfc_eval_summary()
        return "\n".join(
            [
                "Dataset: {} | Selected: {}".format(
                    self._dataset_display_name(self.active_dataset_name),
                    self._dataset_display_name(self.selected_dataset_name),
                ),
                "Rec: {} | Selected: {}".format(self.active_recording_id or "-", self.selected_recording_id or "-"),
                "Frame: {}/{} | {} @ {} fps".format(
                    int(self.current_frame), int(self.maximum_frames), state, int(self.playback_fps)
                ),
                "Visible: {} | Cutter/Neighbor: {}/{}".format(
                    stats.get("visible", self._count_visible_vehicles()),
                    stats.get("cutters", 0),
                    stats.get("neighbors", 0),
                ),
                "Risk C/E/N: {}/{}/{}".format(
                    stats.get("critical", 0),
                    stats.get("elevated", 0),
                    stats.get("normal", 0),
                ),
                "SFC Eval: {}".format(eval_summary),
            ]
        )

    def _count_visible_vehicles(self):
        visible = 0
        frame = int(self.current_frame)
        for track_id, track_info in self.static_info.items():
            initial_frame = int(track_info[INITIAL_FRAME])
            final_frame = int(track_info[FINAL_FRAME])
            if initial_frame <= frame < final_frame:
                visible += 1
        return visible

    def update_slider(self, value):
        if not self.changed_button:
            if self.playing:
                self._toggle_playback(force_stop=True)
            self.current_frame = int(value)
            self.remove_patches()
            self.update_figure()
            self.fig.canvas.draw_idle()
        self.changed_button = False

    def update_speed_slider(self, value):
        self.playback_fps = int(round(value))
        self._apply_playback_interval(immediate=True)
        self._refresh_control_value_labels()
        self._update_header_state()
        self.fig.canvas.draw_idle()

    def update_button_play_pause(self, _):
        self._toggle_playback(force_stop=False)

    def update_button_next(self, _):
        if self.playing:
            self._toggle_playback(force_stop=True)
        if self.current_frame < self.maximum_frames:
            self.current_frame += 1
            self.changed_button = True
            self.trigger_update()
        else:
            print("There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_next2(self, _):
        if self.playing:
            self._toggle_playback(force_stop=True)
        if self.current_frame + 10 <= self.maximum_frames:
            self.current_frame += 10
            self.changed_button = True
            self.trigger_update()
        else:
            self.current_frame = self.maximum_frames
            self.changed_button = True
            self.trigger_update()

    def update_button_previous(self, _):
        if self.playing:
            self._toggle_playback(force_stop=True)
        if self.current_frame > 1:
            self.current_frame -= 1
            self.changed_button = True
            self.trigger_update()
        else:
            print("There are no frames available with an index lower than 1.")

    def update_button_previous2(self, _):
        if self.playing:
            self._toggle_playback(force_stop=True)
        if self.current_frame - 10 > 0:
            self.current_frame -= 10
            self.changed_button = True
            self.trigger_update()
        else:
            self.current_frame = 1
            self.changed_button = True
            self.trigger_update()

    def update_recording_slider(self, value):
        if self.recording_slider is None or not self.recording_options:
            return
        slider_index = int(round(value))
        slider_index = max(1, min(slider_index, len(self.recording_options)))
        self.selected_recording_id = self.recording_options[slider_index - 1]
        self._refresh_control_value_labels()
        self.fig.canvas.draw_idle()

    def update_button_load_recording(self, _):
        if self.dataset_loader is None and self.recording_loader is None:
            print("Recording loader is not configured.")
            return
        if self.selected_recording_id is None:
            print("No recording selected.")
            return
        try:
            if self.dataset_loader is not None:
                loaded = self.dataset_loader(self.selected_dataset_name, self.selected_recording_id)
            else:
                loaded = self.recording_loader(self.selected_recording_id)
            loaded_arguments, loaded_tracks, loaded_static_info, loaded_meta_dictionary = loaded
        except Exception as exc:
            print(
                "Failed to load {} recording {}: {}".format(
                    self._dataset_display_name(self.selected_dataset_name),
                    self.selected_recording_id,
                    exc,
                )
            )
            return
        print(
            "Loaded {} recording {}.".format(
                self._dataset_display_name(self.selected_dataset_name),
                self.selected_recording_id,
            )
        )
        self._reload_recording(loaded_arguments, loaded_tracks, loaded_static_info, loaded_meta_dictionary)

    def on_key_press(self, event):
        if event is None or event.key is None:
            return
        key = event.key.lower()
        if key == " ":
            self.update_button_play_pause(None)
        elif key == "right":
            self.update_button_next(None)
        elif key == "left":
            self.update_button_previous(None)
        elif key == "up":
            self.update_button_next2(None)
        elif key == "down":
            self.update_button_previous2(None)
        elif key == "l":
            self.update_button_load_recording(None)

    def on_mouse_press(self, event):
        if event is None:
            return
        if event.inaxes != self.ax_timeline or event.xdata is None:
            return
        target_frame = int(round(float(event.xdata)))
        target_frame = int(max(1, min(target_frame, self.maximum_frames)))
        if self.playing:
            self._toggle_playback(force_stop=True)
        self.current_frame = target_frame
        self.changed_button = True
        self.trigger_update()

    def on_scroll(self, event):
        if event is None:
            return
        if event.inaxes != self.ax_sfc_info:
            return

        evaluations = self._build_sfc_frame_evaluations()
        direction = str(getattr(event, "button", "")).lower()
        self._set_sfc_eval_count(len(evaluations))
        self._step_sfc_eval(-1 if direction == "up" else +1)

    def trigger_update(self):
        self.remove_patches()
        self.current_frame = int(max(1, min(self.current_frame, self.maximum_frames)))
        self.changed_button = True
        self.frame_slider.set_val(self.current_frame)
        self.changed_button = False
        self.update_figure()
        self._refresh_control_value_labels()
        self.fig.canvas.draw_idle()

    def update_figure(self):
        frame_entries = self.sfc_codes_by_frame.get(int(self.current_frame), [])
        active_cutter_ids = {
            entry.get("cutter_id") for entry in frame_entries if entry.get("cutter_id") is not None
        }
        active_neighbor_ids = self._collect_active_neighbor_ids(active_cutter_ids)

        rect_style_base = dict(fill=True, edgecolor="#e2e8f0", linewidth=0.5, alpha=0.75, zorder=19)
        triangle_style = dict(facecolor="#f8fafc", fill=True, edgecolor="k", lw=0.1, alpha=0.55, zorder=19)
        text_style = dict(picker=True, size=8, color=self.THEME["text_main"], zorder=10, ha="center")
        text_box_style = dict(boxstyle="round,pad=0.2", fc="#0b1b2d", alpha=0.85, ec="#93c5fd", lw=0.5)
        id_on_vehicle_text_style = dict(picker=True, size=7, color="white", zorder=20, ha="center", va="center")
        track_style = dict(color="#94a3b8", linewidth=1.0, zorder=10, alpha=0.6)

        visible_count = 0
        critical_count = 0
        elevated_count = 0
        normal_count = 0
        plotted_objects = []
        for track in self.tracks:
            track_id = self._track_id_as_int(track)
            static_track_information = self.static_info[track_id]
            initial_frame = static_track_information[INITIAL_FRAME]
            final_frame = static_track_information[FINAL_FRAME]
            if not (initial_frame <= self.current_frame < final_frame):
                continue

            current_index = int(self.current_frame - initial_frame)
            risk_class = self._risk_class_for_track(track, current_index)
            if risk_class == "critical":
                critical_count += 1
            elif risk_class == "elevated":
                elevated_count += 1
            else:
                normal_count += 1

            try:
                bounding_box = self._display_bbox(track[BBOX][current_index])
                current_velocity = track[X_VELOCITY][current_index]
                y_position = self.y_sign * bounding_box[1]
                vehicle_box_y = y_position + (self.y_sign * bounding_box[3] if self.y_sign < 0 else 0)
            except Exception:
                continue
            visible_count += 1

            if self.arguments["plotBoundingBoxes"]:
                rect_style = dict(rect_style_base)
                if track_id in active_cutter_ids:
                    rect_style["facecolor"] = self.THEME["cutter"]
                    rect_style["alpha"] = 0.90
                    rect_style["linewidth"] = 1.2
                    rect_style["edgecolor"] = self.THEME["accent"]
                else:
                    rect_style["facecolor"] = self.THEME["danger"]
                    rect_style["alpha"] = 0.76
                    rect_style["linewidth"] = 0.75

                rect = plt.Rectangle((bounding_box[0], vehicle_box_y), bounding_box[2], bounding_box[3], **rect_style)
                self.ax.add_patch(rect)
                plotted_objects.append(rect)

            if self.arguments["plotDirectionTriangle"]:
                triangle_y_position = [y_position, y_position + bounding_box[3], y_position + (bounding_box[3] / 2)]
                if self.y_sign < 0:
                    triangle_y_position += self.y_sign * bounding_box[3]
                if current_velocity < 0:
                    x_back_position = bounding_box[0] + (bounding_box[2] * 0.2)
                    triangle_info = np.array([[x_back_position, x_back_position, bounding_box[0]], triangle_y_position])
                else:
                    x_back_position = bounding_box[0] + bounding_box[2] - (bounding_box[2] * 0.2)
                    triangle_info = np.array(
                        [[x_back_position, x_back_position, bounding_box[0] + bounding_box[2]], triangle_y_position]
                    )
                polygon = plt.Polygon(np.transpose(triangle_info), **triangle_style)
                self.ax.add_patch(polygon)
                plotted_objects.append(polygon)

            if self.arguments["plotTextAnnotation"]:
                vehicle_class = static_track_information[CLASS][0]
                lane_id = int(track[LANE_ID][current_index])
                show_id_on_vehicle = self.arguments.get("plotIdOnlyLabel", False)

                if self.arguments.get("plotDetailedLabel", False):
                    annotation_text = ""
                    if self.arguments["plotClass"]:
                        annotation_text += "{}".format(vehicle_class)
                    if self.arguments["plotVelocity"]:
                        if annotation_text != "":
                            annotation_text += "|"
                        x_velocity = abs(float(current_velocity)) * 3.6
                        annotation_text += "{:.1f}km/h".format(x_velocity)
                    if self.arguments["plotIDs"] and not show_id_on_vehicle:
                        if annotation_text != "":
                            annotation_text += "|"
                        annotation_text += "ID{}".format(track_id)
                    if annotation_text != "":
                        annotation_text += "|"
                    annotation_text += "Lane{}".format(lane_id)
                elif show_id_on_vehicle:
                    annotation_text = "Lane{}".format(lane_id)
                else:
                    annotation_text = "ID{}|Lane{}".format(track_id, lane_id)

                if self.background_image is not None:
                    target_location = (bounding_box[0], y_position - 1)
                    text_location = (bounding_box[0] + (bounding_box[2] / 2), y_position - 1.5)
                else:
                    target_location = (bounding_box[0], y_position + 1)
                    text_location = (bounding_box[0] + (bounding_box[2] / 2), y_position + 1.5)
                text_patch = self.ax.annotate(
                    annotation_text,
                    xy=target_location,
                    xytext=text_location,
                    bbox=text_box_style,
                    **text_style
                )
                text_patch.set_gid("ID{}".format(track_id))
                plotted_objects.append(text_patch)

                if show_id_on_vehicle:
                    vehicle_text_location = (bounding_box[0] + (bounding_box[2] / 2), vehicle_box_y + (bounding_box[3] / 2))
                    id_text_patch = self.ax.annotate(
                        "{}".format(track_id), xy=vehicle_text_location, xytext=vehicle_text_location, **id_on_vehicle_text_style
                    )
                    id_text_patch.set_gid("ID{}".format(track_id))
                    plotted_objects.append(id_text_patch)

            if self.arguments["plotTrackingLines"]:
                relevant_bounding_boxes = np.array(track[BBOX][0:current_index, :], dtype=float)
                if relevant_bounding_boxes.shape[0] > 0:
                    relevant_bounding_boxes = self._display_bbox(relevant_bounding_boxes)
                    sign = 1 if self.background_image is not None else self.y_sign
                    x_centroid_position = relevant_bounding_boxes[:, 0] + relevant_bounding_boxes[:, 2] / 2
                    y_centroid_position = (sign * relevant_bounding_boxes[:, 1]) + sign * (relevant_bounding_boxes[:, 3]) / 2
                    centroids = np.transpose([x_centroid_position, y_centroid_position])
                    track_sign = 1 if current_velocity < 0 else -1
                    plotted_centroids = self.ax.plot(
                        centroids[:, 0] + track_sign * (bounding_box[3] / 2), centroids[:, 1], **track_style
                    )
                    plotted_objects.append(plotted_centroids)

        self.semantic_stats = {
            "visible": int(visible_count),
            "cutters": int(len(active_cutter_ids)),
            "neighbors": int(len(active_neighbor_ids)),
            "critical": int(critical_count),
            "elevated": int(elevated_count),
            "normal": int(normal_count),
        }

        if self.pick_event_cid is not None:
            self.fig.canvas.mpl_disconnect(self.pick_event_cid)
        self.pick_event_cid = self.fig.canvas.mpl_connect("pick_event", self.on_click)
        self.plotted_objects = plotted_objects
        self._update_sfc_matrix_overlay()
        self._refresh_control_value_labels()
        self._update_header_state()
        self._update_timeline_cursor()

    @staticmethod
    def _infer_recording_id(track_csv_path):
        if not track_csv_path:
            return None
        file_name = os.path.basename(track_csv_path)
        match = re.match(r"^(\d+)_tracks\.csv$", file_name)
        if match is None:
            return None
        return int(match.group(1))

    def _load_sfc_codes_by_frame(self, sfc_codes_csv, track_csv_path):
        if not sfc_codes_csv or not os.path.exists(sfc_codes_csv):
            return {}

        usecols = ["event_id", "recording_id", "cutter_id", "stage", "frame", "code", "code_hex"]
        try:
            df = pd.read_csv(sfc_codes_csv, usecols=usecols)
        except Exception:
            return {}

        if "frame" not in df.columns or "code" not in df.columns:
            return {}

        recording_id = self._infer_recording_id(track_csv_path)
        if recording_id is not None and "recording_id" in df.columns:
            rec_values = pd.to_numeric(df["recording_id"], errors="coerce")
            df = df[rec_values == recording_id]

        df = df.copy()
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
        df["code"] = pd.to_numeric(df["code"], errors="coerce")
        df = df.dropna(subset=["frame", "code"])
        if df.empty:
            return {}

        df["frame"] = df["frame"].astype(int)
        df["code"] = df["code"].astype(int)

        codes_by_frame = defaultdict(list)
        for row in df.itertuples(index=False):
            code_hex_value = getattr(row, "code_hex", None)
            if pd.isna(code_hex_value):
                code_hex_value = "{:04x}".format(int(row.code))

            cutter_id_value = getattr(row, "cutter_id", None)
            cutter_id = None
            if cutter_id_value is not None and not pd.isna(cutter_id_value):
                try:
                    cutter_id = int(float(cutter_id_value))
                except Exception:
                    cutter_id = None

            codes_by_frame[int(row.frame)].append(
                {
                    "event_id": getattr(row, "event_id", None),
                    "cutter_id": cutter_id,
                    "stage": getattr(row, "stage", None),
                    "code": int(row.code),
                    "code_hex": str(code_hex_value),
                }
            )
        return dict(codes_by_frame)

    @classmethod
    def _decode_code_to_3x3_matrix(cls, code):
        return [[(int(code) >> cls.HILBERT_BIT_LAYOUT[r][c]) & 1 for c in range(3)] for r in range(3)]

    @staticmethod
    def _parse_neighbor_id(raw_value):
        try:
            value = int(raw_value)
        except Exception:
            try:
                value = int(float(raw_value))
            except Exception:
                return 0
        return value if value > 0 else 0

    def _build_highd_reference_matrix(self, cutter_id, frame):
        try:
            cutter_id_int = int(cutter_id)
        except Exception:
            return None

        track = self.track_lookup.get(cutter_id_int)
        static_track_information = self.static_info.get(cutter_id_int)
        if track is None or static_track_information is None:
            return None

        try:
            initial_frame = int(static_track_information[INITIAL_FRAME])
            final_frame = int(static_track_information[FINAL_FRAME])
        except Exception:
            return None
        if frame < initial_frame or frame >= final_frame:
            return None

        current_index = int(frame - initial_frame)
        g = np.zeros((3, 3), dtype=np.uint8)
        g[1, 1] = 1

        id_columns = (
            (LEFT_PRECEDING_ID, LEFT_ALONGSIDE_ID, LEFT_FOLLOWING_ID),
            (PRECEDING_ID, None, FOLLOWING_ID),
            (RIGHT_PRECEDING_ID, RIGHT_ALONGSIDE_ID, RIGHT_FOLLOWING_ID),
        )
        for col, (preceding_col, alongside_col, following_col) in enumerate(id_columns):
            for row, column_name in ((0, preceding_col), (1, alongside_col), (2, following_col)):
                if column_name is None:
                    continue
                values = track.get(column_name)
                if values is None:
                    continue
                try:
                    raw_value = values[current_index]
                except Exception:
                    continue
                if self._parse_neighbor_id(raw_value) != 0:
                    g[row, col] = 1
        return g.tolist()

    def _build_reference_matrix(self, cutter_id, frame):
        if self._normalize_dataset_name(self.active_dataset_name) == "highd":
            return self._build_highd_reference_matrix(cutter_id, frame)
        return None

    def _reference_matrix_label(self):
        if self._normalize_dataset_name(self.active_dataset_name) == "highd":
            return "highD Raw-ID Matrix"
        return "{} Reference Matrix".format(self._dataset_display_name())

    def _build_sfc_frame_evaluations(self):
        frame = int(self.current_frame)
        frame_entries = self.sfc_codes_by_frame.get(frame, [])
        if not frame_entries:
            return []

        evaluations = []
        sfc_codes_canonical = bool(self.sfc_canonical_effective)
        for entry in frame_entries:
            solution_matrix = self._decode_code_to_3x3_matrix(entry["code"])
            cutter_id = entry.get("cutter_id")
            mirrored = False
            if (not sfc_codes_canonical) and cutter_id is not None and self._is_track_right_to_left(cutter_id, frame):
                solution_matrix = np.fliplr(np.asarray(solution_matrix, dtype=int)).tolist()
                mirrored = True

            reference_matrix = None
            match = None
            if cutter_id is not None:
                reference_matrix = self._build_reference_matrix(cutter_id, frame)
                if reference_matrix is not None:
                    match = bool(
                        np.array_equal(
                            np.asarray(solution_matrix, dtype=int),
                            np.asarray(reference_matrix, dtype=int),
                        )
                    )

            evaluations.append(
                {
                    "entry": entry,
                    "solution_matrix": solution_matrix,
                    "reference_matrix": reference_matrix,
                    "match": match,
                    "mirrored": mirrored,
                }
            )
        return evaluations

    def _build_sfc_eval_summary(self):
        evaluations = self._build_sfc_frame_evaluations()
        if not evaluations:
            return "n/a (no SFC rows)"
        matched = 0
        mismatched = 0
        missing_ref = 0
        for item in evaluations:
            match = item.get("match")
            if match is True:
                matched += 1
            elif match is False:
                mismatched += 1
            else:
                missing_ref += 1
        return "match={}, mismatch={}, missing_ref={}".format(matched, mismatched, missing_ref)

    @staticmethod
    def _select_focus_evaluation(evaluations):
        if not evaluations:
            return None, 0
        for idx, item in enumerate(evaluations):
            if item.get("match") is False:
                return item, idx
        return evaluations[0], 0

    def _style_panel_axes(self, panel):
        panel.set_facecolor(self.THEME["panel_soft"])
        panel.set_xticks([])
        panel.set_yticks([])
        for spine in panel.spines.values():
            spine.set_color(self.THEME["panel_edge"])
            spine.set_linewidth(1.0)

    def _draw_matrix_grid(self, ax, matrix, *, x0, y0, cell, label, palette):
        value_font = self._cell_value_font(ax, cell)
        axis_font = max(5.6, value_font - 1.2)
        title_font = max(6.5, value_font + 0.4)
        ax.text(
            x0,
            y0 + (cell * 3) + 0.04,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=title_font,
            color=self.THEME["text_muted"],
            fontweight="bold",
        )
        # Row labels: P/A/F (preceding/alongside/following), column labels: L/S/R.
        for r, row_name in enumerate(("P", "A", "F")):
            ax.text(
                x0 - 0.018,
                y0 + ((2 - r) * cell) + (cell / 2),
                row_name,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=axis_font,
                color=self.THEME["text_muted"],
            )
        for c, col_name in enumerate(("L", "S", "R")):
            ax.text(
                x0 + (c * cell) + (cell / 2),
                y0 - 0.02,
                col_name,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=axis_font,
                color=self.THEME["text_muted"],
            )

        for r in range(3):
            for c in range(3):
                value = int(matrix[r][c])
                face = palette["on"] if value else palette["off"]
                rect = patches.Rectangle(
                    (x0 + (c * cell), y0 + ((2 - r) * cell)),
                    cell * 0.95,
                    cell * 0.95,
                    transform=ax.transAxes,
                    facecolor=face,
                    edgecolor=self.THEME["panel_edge"],
                    linewidth=0.8,
                    zorder=2,
                    clip_on=False,
                )
                ax.add_patch(rect)
                ax.text(
                    x0 + (c * cell) + (cell * 0.475),
                    y0 + ((2 - r) * cell) + (cell * 0.475),
                    str(value),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=value_font,
                    color="#eaf4ff",
                    fontweight="bold",
                    zorder=3,
                )

    def _draw_diff_grid(self, ax, solution_matrix, reference_matrix, *, x0, y0, cell):
        value_font = self._cell_value_font(ax, cell)
        title_font = max(6.5, value_font + 0.4)
        ax.text(
            x0,
            y0 + (cell * 3) + 0.04,
            "Cell Agreement",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=title_font,
            color=self.THEME["text_muted"],
            fontweight="bold",
        )
        for r in range(3):
            for c in range(3):
                same = int(solution_matrix[r][c]) == int(reference_matrix[r][c])
                face = self.THEME["success"] if same else self.THEME["risk_critical"]
                rect = patches.Rectangle(
                    (x0 + (c * cell), y0 + ((2 - r) * cell)),
                    cell * 0.95,
                    cell * 0.95,
                    transform=ax.transAxes,
                    facecolor=face,
                    edgecolor=self.THEME["panel_edge"],
                    linewidth=0.8,
                    zorder=2,
                    clip_on=False,
                )
                ax.add_patch(rect)
                ax.text(
                    x0 + (c * cell) + (cell * 0.475),
                    y0 + ((2 - r) * cell) + (cell * 0.475),
                    "=" if same else "x",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=value_font,
                    color="#f8fafc",
                    fontweight="bold",
                    zorder=3,
                )

    def _cell_value_font(self, ax, cell):
        """
        Scale matrix cell value text from actual rendered cell size (pixels),
        so text never exceeds the visual cell on small displays.
        """
        try:
            bbox = ax.get_window_extent()
            dpi = float(self.fig.dpi) if self.fig is not None else 100.0
            cell_px = min(float(cell) * float(bbox.width), float(cell) * float(bbox.height))
            pts = (cell_px * 72.0 / max(1.0, dpi)) * 0.68
            return float(max(5.2, min(8.6, pts)))
        except Exception:
            return float(self._responsive_font(7.2))

    def _render_sfc_evaluation_panel(self):
        if not hasattr(self, "ax_sfc_info"):
            return
        self.ax_sfc_info.cla()
        self._style_panel_axes(self.ax_sfc_info)

        evaluations = self._build_sfc_frame_evaluations()
        frame = int(self.current_frame)
        if self.sfc_eval_frame != frame:
            _, default_idx = self._select_focus_evaluation(evaluations)
            self.sfc_eval_index = int(default_idx)
            self.sfc_eval_frame = frame
        self._set_sfc_eval_count(len(evaluations))
        if not evaluations:
            self.ax_sfc_info.text(
                0.02,
                0.7,
                "Frame {} | No SFC rows for this frame".format(frame),
                transform=self.ax_sfc_info.transAxes,
                ha="left",
                va="center",
                fontsize=10.0,
                color=self.THEME["text_main"],
                fontweight="bold",
            )
            self.ax_sfc_info.text(
                0.02,
                0.48,
                "Load a frame with SFC rows to see side-by-side evaluation.",
                transform=self.ax_sfc_info.transAxes,
                ha="left",
                va="center",
                fontsize=8.8,
                color=self.THEME["text_muted"],
            )
            return

        focus_index = int(max(0, min(self.sfc_eval_index, len(evaluations) - 1)))
        focus = evaluations[focus_index]
        entry = focus["entry"]
        solution_matrix = focus["solution_matrix"]
        reference_matrix = focus["reference_matrix"]
        match = focus["match"]
        compact = self._responsive_scale() < 0.90
        title_font = self._responsive_font(9.0)
        meta_font = self._responsive_font(8.0)
        hint_font = self._responsive_font(7.7)
        try:
            axis_h_px = max(1.0, float(self.ax_sfc_info.get_window_extent().height))
            fig_dpi = max(1.0, float(getattr(self.fig, "dpi", 100.0)))
            px_per_pt = fig_dpi / 72.0
            title_step = max(0.075, ((title_font * px_per_pt) * 1.38) / axis_h_px)
            hint_step = max(0.062, ((hint_font * px_per_pt) * 1.30) / axis_h_px)
            meta_step = max(0.058, ((meta_font * px_per_pt) * 1.28) / axis_h_px)
        except Exception:
            title_step = 0.08
            hint_step = 0.064
            meta_step = 0.06
        y_title = 0.965
        y_hint = y_title - title_step
        y_meta_1 = y_hint - hint_step
        y_meta_2 = y_meta_1 - meta_step

        matched = sum(1 for item in evaluations if item.get("match") is True)
        mismatched = sum(1 for item in evaluations if item.get("match") is False)
        missing_ref = sum(1 for item in evaluations if item.get("match") is None)

        self.ax_sfc_info.text(
            0.02,
            y_title,
            "Frame {} | SFC Evaluation Board | showing {}/{}".format(
                frame,
                int(focus_index + 1),
                int(len(evaluations)),
            ),
            transform=self.ax_sfc_info.transAxes,
            ha="left",
            va="top",
            fontsize=title_font,
            color=self.THEME["text_main"],
            fontweight="bold",
            fontfamily="monospace",
        )
        self.ax_sfc_info.text(
            0.02,
            y_hint,
            "Use mouse wheel on this panel to browse SFC rows.",
            transform=self.ax_sfc_info.transAxes,
            ha="left",
            va="top",
            fontsize=hint_font,
            color=self.THEME["text_muted"],
            fontfamily="monospace",
        )
        mode_token = "canonical" if self.sfc_canonical_effective else "raw"
        if self.sfc_canonical_effective != self.sfc_canonical_user:
            mode_token = "{}*".format(mode_token)
        event_line = "event={} cutter={} stage={} mirror={} mode={}".format(
            entry.get("event_id"),
            entry.get("cutter_id"),
            entry.get("stage"),
            "Y" if focus.get("mirrored") else "N",
            mode_token,
        )
        code_line = "code={} hex={} | summary: match={} mismatch={} missing_ref={}".format(
            entry.get("code"),
            entry.get("code_hex"),
            int(matched),
            int(mismatched),
            int(missing_ref),
        )
        if compact:
            self.ax_sfc_info.text(
                0.02,
                y_meta_1,
                self._ellipsize(event_line, 92),
                transform=self.ax_sfc_info.transAxes,
                ha="left",
                va="top",
                fontsize=meta_font,
                color=self.THEME["text_muted"],
                fontfamily="monospace",
            )
            self.ax_sfc_info.text(
                0.02,
                y_meta_2,
                self._ellipsize(code_line, 108),
                transform=self.ax_sfc_info.transAxes,
                ha="left",
                va="top",
                fontsize=meta_font,
                color=self.THEME["text_muted"],
                fontfamily="monospace",
            )
            if self.sfc_orientation_note:
                self.ax_sfc_info.text(
                    0.02,
                    max(0.01, y_meta_2 - meta_step),
                    self._ellipsize(self.sfc_orientation_note, 108),
                    transform=self.ax_sfc_info.transAxes,
                    ha="left",
                    va="top",
                    fontsize=max(6.2, meta_font - 0.2),
                    color="#fbbf24",
                    fontfamily="monospace",
                )
        else:
            self.ax_sfc_info.text(
                0.02,
                y_meta_1,
                self._ellipsize("{} | {}".format(event_line, code_line), 180),
                transform=self.ax_sfc_info.transAxes,
                ha="left",
                va="top",
                fontsize=meta_font,
                color=self.THEME["text_muted"],
                fontfamily="monospace",
            )
            if self.sfc_orientation_note:
                self.ax_sfc_info.text(
                    0.02,
                    max(0.01, y_meta_1 - meta_step),
                    self._ellipsize(self.sfc_orientation_note, 180),
                    transform=self.ax_sfc_info.transAxes,
                    ha="left",
                    va="top",
                    fontsize=max(6.2, meta_font - 0.2),
                    color="#fbbf24",
                    fontfamily="monospace",
                )

        cell = 0.075 if not compact else 0.069
        y0 = 0.29 if not compact else 0.18
        eval_y = 0.18 if not compact else 0.10
        eval_font = self._responsive_font(9.8 if not compact else 8.4)
        self._draw_matrix_grid(
            self.ax_sfc_info,
            solution_matrix,
            x0=0.05,
            y0=y0,
            cell=cell,
            label="Our Matrix (SFC decoded)",
            palette={"on": "#0891b2", "off": "#12263a"},
        )

        if reference_matrix is not None:
            self._draw_matrix_grid(
                self.ax_sfc_info,
                reference_matrix,
                x0=0.36,
                y0=y0,
                cell=cell,
                label=self._reference_matrix_label(),
                palette={"on": "#2563eb", "off": "#12263a"},
            )
            self._draw_diff_grid(self.ax_sfc_info, solution_matrix, reference_matrix, x0=0.67, y0=y0, cell=cell)
            decision_text = "MATCH" if match else "MISMATCH"
            decision_color = self.THEME["success"] if match else self.THEME["risk_critical"]
            self.ax_sfc_info.text(
                0.67,
                eval_y,
                "Evaluation: {}".format(decision_text),
                transform=self.ax_sfc_info.transAxes,
                ha="left",
                va="center",
                fontsize=eval_font,
                color=decision_color,
                fontweight="bold",
            )
        else:
            self.ax_sfc_info.text(
                0.36,
                0.47,
                "{} unavailable".format(self._reference_matrix_label()),
                transform=self.ax_sfc_info.transAxes,
                ha="left",
                va="center",
                fontsize=self._responsive_font(9.0),
                color=self.THEME["text_muted"],
                fontweight="bold",
            )
            self.ax_sfc_info.text(
                0.36,
                0.36,
                "Evaluation: N/A",
                transform=self.ax_sfc_info.transAxes,
                ha="left",
                va="center",
                fontsize=self._responsive_font(9.5),
                color=self.THEME["text_muted"],
                fontweight="bold",
            )

    def _update_sfc_match_indicator(self):
        if not hasattr(self, "sfc_match_text"):
            return
        evaluations = self._build_sfc_frame_evaluations()
        if not evaluations:
            self.sfc_match_text.set_text("SFC MATRIX CHECK: N/A")
            self.sfc_match_text.set_color(self.THEME["text_muted"])
            return

        has_mismatch = any(item.get("match") is False for item in evaluations)
        has_match = any(item.get("match") is True for item in evaluations)
        has_missing = any(item.get("match") is None for item in evaluations)
        if has_mismatch:
            self.sfc_match_text.set_text("SFC MATRIX CHECK: RED (MISMATCH)")
            self.sfc_match_text.set_color(self.THEME["risk_critical"])
            return
        if has_match and (not has_missing):
            self.sfc_match_text.set_text("SFC MATRIX CHECK: GREEN (MATCH)")
            self.sfc_match_text.set_color(self.THEME["success"])
            return
        if has_match:
            self.sfc_match_text.set_text("SFC MATRIX CHECK: GREEN* (PARTIAL)")
            self.sfc_match_text.set_color(self.THEME["success"])
            return

        self.sfc_match_text.set_text("SFC MATRIX CHECK: N/A")
        self.sfc_match_text.set_color(self.THEME["text_muted"])

    def _is_track_right_to_left(self, track_id, frame):
        """Return True when the given track moves toward decreasing x at this frame."""
        try:
            tid = int(track_id)
        except Exception:
            return False

        track = self.track_lookup.get(tid)
        static_track_information = self.static_info.get(tid)
        if track is None or static_track_information is None:
            return False

        try:
            initial_frame = int(static_track_information[INITIAL_FRAME])
            final_frame = int(static_track_information[FINAL_FRAME])
        except Exception:
            return False
        if frame < initial_frame or frame >= final_frame:
            return False

        idx = int(frame - initial_frame)
        try:
            vx = float(track[X_VELOCITY][idx])
            if np.isfinite(vx):
                return vx < 0.0
        except Exception:
            pass

        # Fallback to per-track drivingDirection convention used in highD.
        try:
            dd = float(static_track_information[DRIVING_DIRECTION])
            return int(dd) == 1
        except Exception:
            return False

    def _build_sfc_matrix_text(self):
        # Long multiline matrix dumps are replaced by a visual evaluation board.
        # Keep this as a compact one-line fallback for any existing callers.
        frame = int(self.current_frame)
        evaluations = self._build_sfc_frame_evaluations()
        if not evaluations:
            return "Frame {} | SFC eval board: no rows".format(frame)
        focus, focus_index = self._select_focus_evaluation(evaluations)
        entry = focus["entry"]
        return (
            "Frame {} | showing {}/{} | event={} cutter={} stage={} code={} | {}".format(
                frame,
                int(focus_index + 1),
                int(len(evaluations)),
                entry.get("event_id"),
                entry.get("cutter_id"),
                entry.get("stage"),
                entry.get("code"),
                self._build_sfc_eval_summary(),
            )
        )

    def _update_sfc_matrix_overlay(self):
        if not hasattr(self, "ax_sfc_info"):
            return
        self._render_sfc_evaluation_panel()
        if hasattr(self, "sfc_info_text"):
            self.sfc_info_text.set_text("")
        self._update_sfc_match_indicator()
        if hasattr(self, "status_text"):
            self.status_text.set_fontsize(self._responsive_font(9.2))
        if hasattr(self, "sfc_match_text"):
            self.sfc_match_text.set_fontsize(self._responsive_font(10.2))
        if hasattr(self, "status_text"):
            self.status_text.set_text(self._build_status_text())
        self._update_header_state()

    def plot_highway(self):
        upper_lanes = self.meta_dictionary.get(UPPER_LANE_MARKINGS)
        lower_lanes = self.meta_dictionary.get(LOWER_LANE_MARKINGS)
        if upper_lanes is None or lower_lanes is None:
            self.ax.set_xlim(0, 400)
            self.ax.set_ylim(40, 0)
            return
        upper_lanes = np.asarray(upper_lanes, dtype=float)
        lower_lanes = np.asarray(lower_lanes, dtype=float)
        if upper_lanes.size < 2 or lower_lanes.size < 2:
            self.ax.set_xlim(0, 400)
            self.ax.set_ylim(40, 0)
            return
        upper_lanes_shape = upper_lanes.shape
        lower_lanes_shape = lower_lanes.shape

        asphalt = patches.Rectangle(
            (0, self.y_sign * lower_lanes[lower_lanes_shape[0] - 1] - 5),
            400,
            lower_lanes[lower_lanes_shape[0] - 1] - upper_lanes[0] + 10,
            color="#111827",
            fill=True,
            alpha=1,
            zorder=5,
        )
        self.ax.add_patch(asphalt)

        upper_outer = patches.Rectangle(
            (0, self.y_sign * upper_lanes[0]), 400, self.outer_line_thickness, color=self.lane_color, fill=True, zorder=6
        )
        self.ax.add_patch(upper_outer)
        for i in range(1, upper_lanes_shape[0] - 1):
            self.ax.plot(
                (0, 400),
                (self.y_sign * upper_lanes[i], self.y_sign * upper_lanes[i]),
                color=self.lane_color,
                linestyle="dashed",
                dashes=(25, 70),
                alpha=0.65,
                zorder=6,
            )
        upper_inner = patches.Rectangle(
            (0, self.y_sign * upper_lanes[upper_lanes_shape[0] - 1]),
            400,
            self.outer_line_thickness,
            color=self.lane_color,
            fill=True,
            zorder=6,
        )
        self.ax.add_patch(upper_inner)

        lower_outer = patches.Rectangle(
            (0, self.y_sign * lower_lanes[0]), 400, self.outer_line_thickness, color=self.lane_color, alpha=1, zorder=6
        )
        self.ax.add_patch(lower_outer)
        for i in range(1, lower_lanes_shape[0] - 1):
            self.ax.plot(
                (0, 400),
                (self.y_sign * lower_lanes[i], self.y_sign * lower_lanes[i]),
                color=self.lane_color,
                linestyle="dashed",
                dashes=(25, 70),
                alpha=0.65,
                zorder=6,
            )
        lower_inner = patches.Rectangle(
            (0, self.y_sign * lower_lanes[lower_lanes_shape[0] - 1]),
            400,
            self.outer_line_thickness,
            color=self.lane_color,
            alpha=1,
            zorder=6,
        )
        self.ax.add_patch(lower_inner)

    def plot_highway_information(self):
        # Direction legend is rendered in the header panel to avoid cluttering lane view.
        return

    def on_click(self, event):
        artist = event.artist
        try:
            track_id = None
            artist_gid = artist.get_gid()
            if artist_gid is not None and artist_gid.startswith("ID"):
                track_id = int(artist_gid[2:])
            else:
                text_value = artist._text
                id_start_index = text_value.find("ID")
                if id_start_index == -1:
                    print("Selected label has no vehicle ID. Enable IDs to open vehicle details.")
                    return
                track_id = int(text_value[id_start_index + 2:].split("|")[0])

            selected_track = self.track_lookup.get(track_id)
            if selected_track is None:
                print("No track with the ID {} was found. Nothing to show.".format(track_id))
                return

            static_information = self.static_info[track_id]
            bounding_box = selected_track[BBOX]
            centroids = [bounding_box[:, 0] + bounding_box[:, 2] / 2, bounding_box[:, 1] + bounding_box[:, 3] / 2]
            centroids = np.transpose(centroids)
            initial_frame = static_information[INITIAL_FRAME]
            final_frame = static_information[FINAL_FRAME]
            x_limits = [initial_frame, final_frame]
            track_frames = np.linspace(initial_frame, final_frame, centroids.shape[0], dtype=np.int64)

            fig = plt.figure(np.random.randint(0, 5000, 1))
            try:
                fig.canvas.manager.set_window_title("Track {}".format(track_id))
            except Exception:
                pass
            plt.suptitle("Information for track {}.".format(track_id))

            plt.subplot(311, title="X-Position")
            x_positions = centroids[:, 0]
            borders = [np.amin(x_positions), np.amax(x_positions)]
            plt.plot(track_frames, x_positions)
            plt.plot([self.current_frame, self.current_frame], borders, "--r")
            plt.xlim(x_limits)
            offset = (borders[1] - borders[0]) * 0.05
            plt.ylim([borders[0] - offset, borders[1] + offset])
            plt.xlabel("Frame")
            plt.ylabel("X-Position [m]")

            plt.subplot(312, title="Y-Position")
            y_positions = centroids[:, 1]
            borders = [np.amin(y_positions), np.amax(y_positions)]
            plt.plot(track_frames, y_positions)
            plt.plot([self.current_frame, self.current_frame], borders, "--r")
            plt.xlim(x_limits)
            offset = (borders[1] - borders[0]) * 0.05
            plt.ylim([borders[0] - offset, borders[1] + offset])
            plt.xlabel("Frame")
            plt.ylabel("Y-Position [m]")

            plt.subplot(313, title="X-Velocity")
            velocity = abs(selected_track[X_VELOCITY])
            borders = [np.amin(velocity), np.amax(velocity)]
            plt.plot(track_frames, velocity)
            plt.plot([self.current_frame, self.current_frame], borders, "--r")
            plt.xlim(x_limits)
            offset = (borders[1] - borders[0]) * 0.05
            plt.ylim([borders[0] - offset, borders[1] + offset])
            plt.xlabel("Frame")
            plt.ylabel("X-Velocity [m/s]")

            plt.subplots_adjust(wspace=0.1, hspace=1)
            plt.show()
        except Exception:
            print("Something went wrong when trying to plot the detailed information of the vehicle.")
            return

    def get_figure(self):
        return self.fig

    def remove_patches(self):
        if self.pick_event_cid is not None:
            self.fig.canvas.mpl_disconnect(self.pick_event_cid)
            self.pick_event_cid = None
        for figure_object in self.plotted_objects:
            try:
                if isinstance(figure_object, list):
                    figure_object[0].remove()
                else:
                    figure_object.remove()
            except Exception:
                continue
        self.plotted_objects = []

    @staticmethod
    def show():
        plt.show()
        plt.close()
