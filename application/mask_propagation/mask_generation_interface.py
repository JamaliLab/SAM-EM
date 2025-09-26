import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
from PIL import Image
import os

class SegmentationUI:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self._build_ui()

    def _build_ui(self):
        # Paths Frame
        path_frame = ctk.CTkFrame(self.parent)
        path_frame.pack(padx=10, pady=10, fill="x")
        path_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(path_frame, text="Video Directory:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.app.video_dir_entry = ctk.CTkEntry(path_frame)
        if self.app.video_dir: 
            self.app.video_dir_entry.insert(0, self.app.video_dir)
        self.app.video_dir_entry.grid(row=0, column=1, sticky="ew")
        ctk.CTkButton(path_frame, text="Browse...", width=80, command=self.app.select_video_dir).grid(row=0, column=2, padx=10)

        ctk.CTkLabel(path_frame, text="Output Directory:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.app.output_dir_entry = ctk.CTkEntry(path_frame)
        if self.app.output_dir: 
            self.app.output_dir_entry.insert(0, self.app.output_dir)
        self.app.output_dir_entry.grid(row=1, column=1, sticky="ew")
        ctk.CTkButton(path_frame, text="Browse...", width=80, command=self.app.select_output_dir).grid(row=1, column=2, padx=10)

        self.app.init_button = ctk.CTkButton(path_frame, text="Load Video & Initialize", command=self.app.initialize_inference)
        self.app.init_button.grid(row=2, column=0, columnspan=3, pady=10)

        self.app.init_progress_label = ctk.CTkLabel(path_frame, text="")
        self.app.init_progress_label.grid(row=3, column=0, columnspan=3, sticky="ew")
        self.app.init_progress = ctk.CTkProgressBar(path_frame, orientation="horizontal", mode='indeterminate')

        # Segmentation Frame
        seg_frame = ctk.CTkFrame(self.parent)
        seg_frame.pack(padx=10, pady=5, fill="x")
        seg_frame.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(seg_frame, text="Frame to Annotate:").grid(row=0, column=0, padx=10)
        self.app.frame_idx_entry = ctk.CTkEntry(seg_frame, width=120)
        self.app.frame_idx_entry.grid(row=0, column=1, padx=(0,10))

        ctk.CTkButton(seg_frame, text="Annotate Frame", command=self.app.annotate_frame).grid(row=0, column=2, pady=10, padx=(0,5))
        ctk.CTkButton(seg_frame, text="Reset All Predictions", command=self.app.reset_predictions).grid(row=0, column=3, pady=10)

        # Propagation Frame
        prop_frame = ctk.CTkFrame(self.parent)
        prop_frame.pack(padx=10, pady=5, fill="x", expand=True)

        self.app.prop_button = ctk.CTkButton(prop_frame, text="Propagate Masks Through Video", command=self.app.propagate_masks)
        self.app.prop_button.pack(pady=10)

        self.app.prop_progress_label = ctk.CTkLabel(prop_frame, text="")
        self.app.prop_progress_label.pack(fill="x", padx=10)
        self.app.prop_progress = ctk.CTkProgressBar(prop_frame, orientation="horizontal", mode='determinate')
        self.app.prop_progress.set(0)
