"""Matplotlib-based interactive frame viewer for highD tracks and SFC context."""

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


class VisualizationPlot(object):
    """Interactive UI for stepping through frames and inspecting vehicle context."""

    HILBERT_BIT_LAYOUT = (
        (0, 1, 14, 15),
        (3, 2, 13, 12),
        (4, 7, 8, 11),
        (5, 6, 9, 10),
    )

    THEME = {
        "figure_bg": "#071521",
        "road_bg": "#0b1220",
        "panel": "#0e2031",
        "panel_soft": "#11283d",
        "panel_edge": "#27445f",
        "text_main": "#e5f0ff",
        "text_muted": "#9eb7d4",
        "button": "#173650",
        "button_hover": "#24547c",
        "button_text": "#eaf4ff",
        "accent": "#00d4c0",
        "accent_alt": "#38bdf8",
        "danger": "#fb7185",
        "cutter": "#f59e0b",
        "cutter_trail": "#7dd3fc",
        "neighbor": "#22d3ee",
        "risk_critical": "#ef4444",
        "risk_elevated": "#f59e0b",
        "risk_normal": "#fb7185",
    }

    def __init__(
        self,
        arguments,
        read_tracks,
        static_info,
        meta_dictionary,
        recording_loader=None,
        recording_options=None,
        fig=None,
    ):
        self.arguments = arguments
        self.tracks = read_tracks
        self.static_info = static_info
        self.meta_dictionary = meta_dictionary
        self.recording_loader = recording_loader
        self.recording_options = self._normalize_recording_options(recording_options)
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
        self.playing = False
        self.playback_fps = 12
        self.playback_timer = None
        self.default_xlim = None
        self.default_ylim = None
        self.timeline_event_points = []
        self.timeline_seek_line = None
        self.timeline_stage_colors = {
            "decision": "#22d3ee",
            "execution": "#f59e0b",
            "merge": "#a78bfa",
            "other": "#94a3b8",
        }
        self.sfc_codes_by_frame = self._load_sfc_codes_by_frame(
            arguments.get("sfc_codes_csv"),
            arguments.get("input_path"),
        )
        self.timeline_event_points = self._build_timeline_event_points()
        self.semantic_stats = {
            "visible": 0,
            "cutters": 0,
            "neighbors": 0,
            "critical": 0,
            "elevated": 0,
            "normal": 0,
        }

        self.road_bottom = 0.36
        self.road_top = 0.86
        self.header_bottom = 0.885
        self.header_height = 0.075
        self.info_bottom = 0.225
        self.info_height = 0.105
        self.timeline_bottom = 0.145
        self.timeline_height = 0.03
        self.controls_bottom = 0.055
        self.controls_height = 0.065

        if fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self.fig.set_size_inches(24, 9)
        else:
            self.fig = fig
            self.ax = self.fig.gca()

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

        self.ax_header.text(
            0.015,
            0.62,
            "highD Command Deck",
            transform=self.ax_header.transAxes,
            ha="left",
            va="center",
            fontsize=15,
            color=self.THEME["text_main"],
            fontweight="bold",
        )
        self.ax_header.text(
            0.015,
            0.22,
            "Controls: Space play/pause | Left/Right +/-1 | Up/Down +/-10 | L load recording",
            transform=self.ax_header.transAxes,
            ha="left",
            va="center",
            fontsize=9,
            color=self.THEME["text_muted"],
        )
        self.header_state_text = self.ax_header.text(
            0.985,
            0.62,
            "",
            transform=self.ax_header.transAxes,
            ha="right",
            va="center",
            fontsize=9.5,
            color=self.THEME["text_main"],
            fontweight="bold",
        )

    def _build_timeline(self):
        self.ax_timeline = self.fig.add_axes([0.02, 0.338, 0.96, 0.018])
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
        self.ax_sfc_info = self.fig.add_axes([0.02, self.info_bottom, 0.66, self.info_height])
        self.ax_status = self.fig.add_axes([0.70, self.info_bottom, 0.28, self.info_height])

        for panel in (self.ax_sfc_info, self.ax_status):
            panel.set_facecolor(self.THEME["panel_soft"])
            panel.set_xticks([])
            panel.set_yticks([])
            for spine in panel.spines.values():
                spine.set_color(self.THEME["panel_edge"])
                spine.set_linewidth(1.0)

        self.sfc_info_text = self.ax_sfc_info.text(
            0.02,
            0.95,
            "",
            transform=self.ax_sfc_info.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
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
            fontsize=8.2,
            color=self.THEME["text_main"],
            linespacing=1.25,
        )
        self._draw_status_legend()

    def _build_controls(self):
        control_gap = 0.008
        button_height = 0.05

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
                edgecolor="#dbeafe",
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
                fontsize=6.6,
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
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )

        for frame, stage in self.timeline_event_points:
            color = self.timeline_stage_colors.get(stage, self.timeline_stage_colors["other"])
            self.ax_timeline.vlines(
                frame,
                0.15,
                0.85,
                color=color,
                linewidth=1.0,
                alpha=0.85,
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

        if self.key_event_cid is not None:
            self.fig.canvas.mpl_disconnect(self.key_event_cid)
        self.key_event_cid = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        if self.mouse_event_cid is not None:
            self.fig.canvas.mpl_disconnect(self.mouse_event_cid)
        self.mouse_event_cid = self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)

    def _style_button(self, button):
        button.ax.set_facecolor(self.THEME["button"])
        button.color = self.THEME["button"]
        button.hovercolor = self.THEME["button_hover"]
        button.label.set_color(self.THEME["button_text"])
        button.label.set_fontsize(9)
        for spine in button.ax.spines.values():
            spine.set_color(self.THEME["panel_edge"])
            spine.set_linewidth(0.9)

    def _style_slider(self, slider):
        slider.poly.set_facecolor(self.THEME["accent"])
        slider.vline.set_color(self.THEME["accent_alt"])
        slider.label.set_visible(False)
        slider.valtext.set_color(self.THEME["text_main"])
        slider.valtext.set_transform(slider.ax.transAxes)
        slider.valtext.set_position((0.988, 0.5))
        slider.valtext.set_ha("right")
        slider.valtext.set_va("center")
        slider.valtext.set_fontsize(8.5)
        slider.valtext.set_bbox(
            dict(boxstyle="round,pad=0.16", fc=self.THEME["panel"], ec=self.THEME["panel_edge"], lw=0.7, alpha=0.92)
        )
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
            fontsize=8.5,
            color=self.THEME["text_muted"],
            fontweight="bold",
            clip_on=False,
        )

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
                fontsize=9,
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
            "REC {} | FRAME {}/{} | {} @ {}fps".format(
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
            self.ax.imshow(self.background_image[:, :, :], alpha=0.95)
            tint = np.zeros_like(self.background_image[:, :, :], dtype=float)
            self.ax.imshow(tint, alpha=0.08)
        else:
            self.background_image = None
            self.y_sign = -1
            self.outer_line_thickness = 0.2
            self.lane_color = "#b0c7dc"
            self.plot_highway()

        self.plot_highway_information()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_autoscale_on(False)
        self.default_xlim = tuple(self.ax.get_xlim())
        self.default_ylim = tuple(self.ax.get_ylim())

    def _reload_recording(self, loaded_arguments, loaded_tracks, loaded_static_info, loaded_meta_dictionary):
        self._toggle_playback(force_stop=True)
        self.remove_patches()
        self.arguments = loaded_arguments
        self.tracks = loaded_tracks
        self.static_info = loaded_static_info
        self.meta_dictionary = loaded_meta_dictionary
        self.track_lookup = self._build_track_lookup(self.tracks)
        self.active_recording_id = self._resolve_recording_id(loaded_arguments)
        self.selected_recording_id = self.active_recording_id
        self.maximum_frames = self._compute_maximum_frame()
        self.current_frame = 1
        self.sfc_codes_by_frame = self._load_sfc_codes_by_frame(
            loaded_arguments.get("sfc_codes_csv"),
            loaded_arguments.get("input_path"),
        )
        self.timeline_event_points = self._build_timeline_event_points()
        self._refresh_timeline_plot()
        self._draw_background()

        self.frame_slider = self._build_frame_slider()
        self.frame_slider.on_changed(self.update_slider)

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

        bounding_box = np.array(track[BBOX][current_index], dtype=float)
        if self.background_image is not None:
            bounding_box /= 0.10106
            bounding_box /= 4

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
        return "\n".join(
            [
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
        if self.playback_timer is not None:
            self.playback_timer.interval = self._fps_to_interval(self.playback_fps)
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
        if self.recording_loader is None:
            print("Recording loader is not configured.")
            return
        if self.selected_recording_id is None:
            print("No recording selected.")
            return
        try:
            loaded = self.recording_loader(self.selected_recording_id)
            loaded_arguments, loaded_tracks, loaded_static_info, loaded_meta_dictionary = loaded
        except Exception as exc:
            print("Failed to load recording {}: {}".format(self.selected_recording_id, exc))
            return
        print("Loaded recording {}.".format(self.selected_recording_id))
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
                bounding_box = np.array(track[BBOX][current_index], dtype=float)
                current_velocity = track[X_VELOCITY][current_index]
                if self.background_image is not None:
                    bounding_box /= 0.10106
                    bounding_box /= 4
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
                    if self.background_image is not None:
                        relevant_bounding_boxes /= 0.10106
                        relevant_bounding_boxes /= 4
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

    def _build_sfc_matrix_text(self):
        frame = int(self.current_frame)
        frame_entries = self.sfc_codes_by_frame.get(frame)
        if not frame_entries:
            return "Frame {} SFC matrix: no code".format(frame)

        lines = ["Frame {} SFC 3x3 matrix/matrices:".format(frame)]
        for entry in frame_entries:
            matrix = self._decode_code_to_3x3_matrix(entry["code"])
            lines.append(
                "event_id={}, cutter_id={}, stage={}, code={}, code_hex={}".format(
                    entry.get("event_id"),
                    entry.get("cutter_id"),
                    entry.get("stage"),
                    entry["code"],
                    entry.get("code_hex"),
                )
            )
            lines.extend(" ".join(str(cell) for cell in row) for row in matrix)
            lines.append("")
        return "\n".join(lines).rstrip()

    def _update_sfc_matrix_overlay(self):
        if not hasattr(self, "sfc_info_text"):
            return
        self.sfc_info_text.set_text(self._build_sfc_matrix_text())
        if hasattr(self, "status_text"):
            self.status_text.set_text(self._build_status_text())
        self._update_header_state()

    def plot_highway(self):
        upper_lanes = self.meta_dictionary[UPPER_LANE_MARKINGS]
        upper_lanes_shape = upper_lanes.shape
        lower_lanes = self.meta_dictionary[LOWER_LANE_MARKINGS]
        lower_lanes_shape = lower_lanes.shape

        asphalt = patches.Rectangle(
            (0, self.y_sign * lower_lanes[lower_lanes_shape[0] - 1] - 5),
            400,
            lower_lanes[lower_lanes_shape[0] - 1] - upper_lanes[0] + 10,
            color="#1e293b",
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
                alpha=0.8,
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
                alpha=0.8,
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
        info_box_style = dict(boxstyle="round,pad=0.22", fc="#0b1b2d", alpha=0.80, ec="#60a5fa", lw=0.6)
        self.ax.text(
            0.5,
            0.985,
            "Direction 1: X-velocity negative (upper lanes)",
            transform=self.ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color=self.THEME["text_main"],
            bbox=info_box_style,
            zorder=30,
        )
        self.ax.text(
            0.5,
            0.015,
            "Direction 2: X-velocity positive (lower lanes)",
            transform=self.ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color=self.THEME["text_main"],
            bbox=info_box_style,
            zorder=30,
        )

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
