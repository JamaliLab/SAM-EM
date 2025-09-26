import os
import sys
import threading
import customtkinter as ctk
from PIL import Image
from tkinter import filedialog, colorchooser
from CTkMessagebox import CTkMessagebox
from CTkColorPicker import AskColor
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import ListedColormap
import torch

from mask_propagation.mask_component import MaskComponent
from mask_propagation.mask_generation_interface import SegmentationUI
from particle_tracking.particle_tracking_interface import ParticleTrackingUI

def show_mask(mask, ax, obj_id=None, random_color=False, cmap="Paired", alpha=0.7):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        cmap_instance = plt.get_cmap(cmap)
        color = np.array([*cmap_instance(obj_id if obj_id is not None else 0)[:3], alpha])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master.root)
        self.master_app = master
        self.title("Settings")
        self.transient(master.root)
        self.grab_set()

        self.model_checkpoint_var = ctk.StringVar(value=self.master_app.model_checkpoint)
        self.model_cfg_var = ctk.StringVar(value=self.master_app.model_cfg)
        self.point_color_var = ctk.StringVar(value=self.master_app.point_color)
        self.box_color_var = ctk.StringVar(value=self.master_app.box_color)

        self._setup_widgets()

    def _setup_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        # Model paths
        path_frame = ctk.CTkFrame(main_frame)
        path_frame.pack(fill=ctk.X, pady=5)
        path_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(path_frame, text="Checkpoint (.pt):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(path_frame, textvariable=self.model_checkpoint_var).grid(row=0, column=1, sticky="ew", padx=5)
        ctk.CTkButton(path_frame, text="...", width=30, command=lambda: self.browse_file(self.model_checkpoint_var)).grid(row=0, column=2, padx=5)

        ctk.CTkLabel(path_frame, text="Config (.yaml):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(path_frame, textvariable=self.model_cfg_var).grid(row=1, column=1, sticky="ew", padx=5)
        ctk.CTkButton(path_frame, text="...", width=30, command=lambda: self.browse_file(self.model_cfg_var)).grid(row=1, column=2, padx=5)

        # Prompt colors
        color_frame = ctk.CTkFrame(main_frame)
        color_frame.pack(fill=ctk.X, pady=5)

        ctk.CTkLabel(color_frame, text="Point Color:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.point_color_label = ctk.CTkLabel(color_frame, text="", fg_color=self.point_color_var.get(), width=50)
        self.point_color_label.grid(row=0, column=1, sticky="w")
        ctk.CTkButton(color_frame, text="Choose...", command=lambda: self.choose_color(self.point_color_var, self.point_color_label)).grid(row=0, column=2, padx=10)

        ctk.CTkLabel(color_frame, text="Box Color:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.box_color_label = ctk.CTkLabel(color_frame, text="", fg_color=self.box_color_var.get(), width=50)
        self.box_color_label.grid(row=1, column=1, sticky="w")
        ctk.CTkButton(color_frame, text="Choose...", command=lambda: self.choose_color(self.box_color_var, self.box_color_label)).grid(row=1, column=2, padx=10)

        # Actions
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill=ctk.X, pady=(10, 0))
        ctk.CTkButton(button_frame, text="Cancel", command=self.destroy).pack(side=ctk.RIGHT, padx=5)
        ctk.CTkButton(button_frame, text="Save & Close", command=self.save_and_close).pack(side=ctk.RIGHT)

    def browse_file(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def choose_color(self, var, label):
        picker = AskColor()  
        color = picker.get() 
        if color:
            var.set(color)
            label.configure(fg_color=color)

    def save_and_close(self):
        self.master_app.model_checkpoint = self.model_checkpoint_var.get()
        self.master_app.model_cfg = self.model_cfg_var.get()
        self.master_app.point_color = self.point_color_var.get()
        self.master_app.box_color = self.box_color_var.get()

        self.master_app._save_settings()

        CTkMessagebox(
            title="Settings Saved",
            message="Settings updated. Re-initialize to apply model path changes.",
            icon="check"
        )
        self.after(100, self.destroy)


class PromptWindow(ctk.CTkToplevel):
    def __init__(self, master, frame_idx, image):
        super().__init__(master.root)
        self.master_app = master
        self.frame_idx = frame_idx
        self.image = image
        self.title(f"Annotating Frame {frame_idx}")
        self.geometry("800x600")

        self.point_coords = []
        self.box_coords = None
        self.drawn_patches = []

        self.point_color = self.master_app.point_color
        self.box_color = self.master_app.box_color
        self.rect_selector = None
        self.point_cid = None

        self._setup_widgets()
        self._setup_canvas()
        self.switch_prompt_mode()

    def _setup_widgets(self):
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(side=ctk.TOP, fill=ctk.X, padx=10, pady=5)

        ctk.CTkLabel(top_frame, text="Particle ID:").pack(side=ctk.LEFT)
        self.particle_id_entry = ctk.CTkEntry(top_frame, width=100)
        self.particle_id_entry.pack(side=ctk.LEFT, padx=5)

        self.prompt_type = ctk.StringVar(value="box")
        ctk.CTkRadioButton(top_frame, text="Box Prompt", variable=self.prompt_type, value="box",
                           command=self.switch_prompt_mode).pack(side=ctk.LEFT, padx=5)
        ctk.CTkRadioButton(top_frame, text="Point Prompt", variable=self.prompt_type, value="points",
                           command=self.switch_prompt_mode).pack(side=ctk.LEFT, padx=5)

        self.done_button = ctk.CTkButton(top_frame, text="Generate Mask", command=self.on_done_click)
        self.done_button.pack(side=ctk.LEFT, padx=5)

        ctk.CTkButton(top_frame, text="Close & Save Prompts", command=self.destroy).pack(side=ctk.RIGHT, padx=5)

    def _setup_canvas(self):
        # Frame to hold the matplotlib canvas
        canvas_frame = ctk.CTkFrame(self, fg_color="transparent")
        canvas_frame.pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)

        fig, self.ax = plt.subplots()
        fig.patch.set_facecolor("#2b2b2b")   # figure background
        self.ax.set_facecolor("#2b2b2b")    # axes background
        self.ax.tick_params(colors="white") # make tick labels visible
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")
        self.ax.imshow(self.image, cmap='gray')
        fig.tight_layout() # Adjust plot to fit into the figure area.

        self.canvas_widget = FigureCanvasTkAgg(fig, master=canvas_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(side=ctk.BOTTOM, fill=ctk.X, padx=10, pady=5)

        ctk.CTkLabel(bottom_frame, text="X:").pack(side=ctk.LEFT, padx=(5,0))
        self.x_entry = ctk.CTkEntry(bottom_frame, width=80)
        self.x_entry.pack(side=ctk.LEFT, padx=(0,5))
        ctk.CTkLabel(bottom_frame, text="Y:").pack(side=ctk.LEFT, padx=(5,0))
        self.y_entry = ctk.CTkEntry(bottom_frame, width=80)
        self.y_entry.pack(side=ctk.LEFT, padx=(0,5))
        ctk.CTkButton(bottom_frame, text="Add Point", command=self.add_manual_coord).pack(side=ctk.LEFT, padx=5)
        ctk.CTkButton(bottom_frame, text="Clear Last", command=self.clear_last_prompt).pack(side=ctk.LEFT, padx=5)

    def switch_prompt_mode(self):
        if self.prompt_type.get() == 'points':
            if self.rect_selector:
                self.rect_selector.set_active(False)
            self.point_cid = self.canvas_widget.figure.canvas.mpl_connect('button_press_event', self.on_point_click)
        else:
            if self.point_cid:
                self.canvas_widget.figure.canvas.mpl_disconnect(self.point_cid)
                self.point_cid = None
            if not self.rect_selector:
                self.rect_selector = RectangleSelector(
                    self.ax, self.on_box_select, useblit=True, button=[1],
                    minspanx=5, minspany=5, spancoords='pixels', interactive=True,
                    props=dict(facecolor='none', edgecolor=self.box_color, linewidth=1.5),
                    handle_props=dict(marker='s', markersize=0, mfc='none', mec='none')
                )
            self.rect_selector.set_active(True)

    def on_point_click(self, event):
        if event.inaxes != self.ax:
            return
        self.point_coords.append((event.xdata, event.ydata))
        point = self.ax.plot(event.xdata, event.ydata, 'x', color=self.point_color, markersize=8)[0]
        self.drawn_patches.append(point)
        self.canvas_widget.draw()

    def on_box_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.box_coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def add_manual_coord(self):
        try:
            x = float(self.x_entry.get()); y = float(self.y_entry.get())
            self.point_coords.append((x, y))
            point = self.ax.plot(x, y, 'x', color=self.point_color, markersize=8)[0]
            self.drawn_patches.append(point)

            if self.prompt_type.get() == 'box' and len(self.point_coords) >= 2:
                p1, p2 = self.point_coords[-2], self.point_coords[-1]
                self.box_coords = [
                    min(p1[0], p2[0]), min(p1[1], p2[1]),
                    max(p1[0], p2[0]), max(p1[1], p2[1])
                ]
                rect = Rectangle(
                    (self.box_coords[0], self.box_coords[1]),
                    self.box_coords[2] - self.box_coords[0],
                    self.box_coords[3] - self.box_coords[1],
                    fill=False, edgecolor=self.box_color, linewidth=1.5
                )
                self.ax.add_patch(rect)
                self.drawn_patches.append(rect)

            self.canvas_widget.draw()
            self.x_entry.delete(0, ctk.END); self.y_entry.delete(0, ctk.END)
            self.x_entry.focus_set()
        except (ValueError, IndexError):
            CTkMessagebox(title="Input Error", message="Please enter valid numeric coordinates.", icon="cancel", parent=self)

    def clear_last_prompt(self):
        if not self.drawn_patches:
            return
        last_patch = self.drawn_patches.pop()
        last_patch.remove()

        if isinstance(last_patch, Rectangle):
            self.box_coords = None
            if self.drawn_patches:
                last_point_patch = self.drawn_patches.pop()
                last_point_patch.remove()
            if self.point_coords:
                self.point_coords.pop()
        elif self.point_coords:
            self.point_coords.pop()

        self.canvas_widget.draw()

    def on_done_click(self):
        try:
            particle_id = int(self.particle_id_entry.get())
        except ValueError:
            CTkMessagebox(title="Error", message="Particle ID must be an integer.", icon="cancel", parent=self)
            return

        self.done_button.configure(text="Generating...", state=ctk.DISABLED)
        self.update_idletasks()

        try:
            if self.prompt_type.get() == "box" and self.box_coords:
                obj_ids, masks = self.master_app.backend.add_box(self.frame_idx, particle_id, self.box_coords)
                self.box_coords = None
                self.point_coords.clear()
            elif self.prompt_type.get() == "points" and self.point_coords:
                points = np.array(self.point_coords, dtype=np.float32)
                labels = np.ones(len(self.point_coords), dtype=np.float32)
                obj_ids, masks = self.master_app.backend.add_points(self.frame_idx, particle_id, points, labels)
                self.point_coords = []
            else:
                CTkMessagebox(title="Warning", message="No prompt to generate mask from.", icon="warning", parent=self)
                self.done_button.configure(text="Generate Mask", state=ctk.NORMAL)
                return
        except Exception as e:
            CTkMessagebox(title="Mask Error", message=f"Failed to generate mask:\n{e}", icon="cancel", parent=self)
            self.done_button.configure(text="Generate Mask", state=ctk.NORMAL)
            return

        # mark that prompts were added
        self.master_app.prompts_added = True

        # clear drawn prompts
        for p in self.drawn_patches:
            p.remove()
        self.drawn_patches.clear()

        if self.rect_selector:
            self.rect_selector.extents = (0, 0, 0, 0)

        # visualize masks
        for obj_id, m in zip(obj_ids, masks):
            show_mask(np.squeeze(m), self.ax, obj_id=obj_id, cmap=self.master_app.cmap_custom)

        self.canvas_widget.draw()
        self.done_button.configure(text="Generate Mask", state=ctk.NORMAL)


class MaskApp:
    def __init__(self, root):
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.title("SAM-EM Mask Generation - Main")
        self.root.geometry("650x500")

        self.APP_DIR = Path(__file__).resolve().parent
        self.SETTINGS_PATH = self.APP_DIR / ".sam_em" / "settings.json"
        self.SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Config defaults (user can change in Settings)
        self.model_checkpoint = "../checkpoints/finetuned_sam2.1.pt"
        self.model_cfg = "../sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        self.point_color = 'red'
        self.box_color = 'green'

        # Runtime
        self.backend: MaskComponent | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_dir = None
        self.output_dir = None
        self.frame_names = []
        self.prompts_added = False
        self._load_settings()
        # Colormap for overlay
        self.colors_custom = [
            "#E69F00", "#D55E00", "#56B4E9", "#009E73",
            "#0072B2", "#CC79A7", "#F0E442", "#B3D100", "#9E1B32"
        ]
        self.cmap_custom = ListedColormap(self.colors_custom, name="cmap_custom")

        self._build_ui()

    def on_close(self):
        self.root.destroy()
        os._exit(0)

    def _load_settings(self):
        """Load settings from JSON if it exists."""
        try:
            if self.SETTINGS_PATH.is_file():
                with open(self.SETTINGS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.model_checkpoint = data.get("model_checkpoint", self.model_checkpoint)
                self.model_cfg = data.get("model_cfg", self.model_cfg)
                self.point_color = data.get("point_color", self.point_color)
                self.box_color = data.get("box_color", self.box_color)
                self.video_dir = data.get("video_dir", self.video_dir)
                self.output_dir = data.get("output_dir", self.output_dir)
        except Exception as e:
            CTkMessagebox(title="Settings", message=f"Could not load settings file:\n{e}", icon="warning")

    def _save_settings(self):
        """Persist current settings to JSON."""
        data = {
            "model_checkpoint": self.model_checkpoint,
            "model_cfg": self.model_cfg,
            "point_color": self.point_color,
            "box_color": self.box_color,
            "video_dir": self.video_dir,
            "output_dir": self.output_dir,
        }
        try:
            with self.SETTINGS_PATH.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            CTkMessagebox(title="Settings", message=f"Could not save settings:\n{e}", icon="warning")

    def _build_ui(self):
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)
        top_bar_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        top_bar_frame.pack(fill="x", padx=10, pady=0)

        icon_path = self.APP_DIR / "images" / "gear.png"
        pil_img = Image.open(icon_path)
        gear_icon = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(20, 20))
        settings_btn = ctk.CTkButton(
            top_bar_frame,
            text="", 
            image=gear_icon,
            command=self.open_settings,
            width=30,   
            height=25,
            corner_radius=6,   
            font=("", 12)       
        )
        settings_btn.pack(side="right", padx=5, pady=5)
        self.tabview = ctk.CTkTabview(main_frame, width=600, height=400)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=0)

        seg_tab = self.tabview.add("Segmentation")
        track_tab = self.tabview.add("Particle Tracking")
        self.tabview.set("Segmentation")

        self.segmentation_ui = SegmentationUI(seg_tab, self)
        self.tracking_ui = ParticleTrackingUI(track_tab, self)
    
    def _validate_model_paths(self) -> bool:
        missing = []
        if not os.path.isfile(self.model_cfg):
            missing.append(f"Config (.yaml) not found:\n{self.model_cfg}")
        if not os.path.isfile(self.model_checkpoint):
            missing.append(f"Checkpoint (.pt) not found:\n{self.model_checkpoint}")

        if missing:
            CTkMessagebox(
                title="Model Files Missing",
                message="Please fix the following before initializing:\n\n" + "\n\n".join(missing) +
                        "\n\nTip: Open Settings to update paths.",
                icon="cancel"
            )
            return False
        return True

    def open_settings(self):
        SettingsWindow(self)

    def select_video_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.video_dir_entry.delete(0, ctk.END)
            self.video_dir_entry.insert(0, dir_path)

    def select_output_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir_entry.delete(0, ctk.END)
            self.output_dir_entry.insert(0, dir_path)

    def initialize_inference(self):
        self.video_dir = self.video_dir_entry.get()
        self.output_dir = self.output_dir_entry.get()
        if not self.video_dir or not os.path.isdir(self.video_dir):
            CTkMessagebox(title="Error", message="Please select a valid video directory.", icon="cancel")
            return
        if not self.output_dir:
             # Make the directory if it doesn't exist
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except OSError as e:
                CTkMessagebox(title="Error", message=f"Could not create output directory:\n{e}", icon="cancel")
                return
        if not self._validate_model_paths():
            return

        self._save_settings()
        self.init_button.configure(state=ctk.DISABLED)
        self.init_progress.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 5), padx=10)
        self.init_progress_label.configure(text="Initializing, please wait...")
        self.init_progress.start()

        threading.Thread(target=self._initialize_worker, daemon=True).start()

    def _initialize_worker(self):
        try:
            self.backend = MaskComponent(self.model_cfg, self.model_checkpoint, device=self.device)
            self.frame_names = self.backend.init_video(self.video_dir)
            self.prompts_added = False
            success = True
            err_msg = ""
        except Exception as e:
            success = False
            err_msg = str(e)

        def finalize():
            self.init_progress.stop()
            self.init_progress.grid_remove()
            self.init_button.configure(state=ctk.NORMAL)
            self.init_progress_label.configure(text="Initialization Complete!" if success else f"Initialization Failed: {err_msg}")
            if success:
                CTkMessagebox(title="Success", message="Inference state initialized.", icon="check")
            else:
                CTkMessagebox(title="Initialization Error", message=err_msg, icon="cancel")

        self.root.after(0, finalize)

    def annotate_frame(self):
        if not self.backend or not self.backend.inference_state:
            CTkMessagebox(title="Error", message="Please initialize inference first.", icon="cancel")
            return

        try:
            frame_idx = int(self.frame_idx_entry.get())
        except ValueError:
            CTkMessagebox(title="Error", message="Frame Index must be an integer.", icon="cancel")
            return

        if not (0 <= frame_idx < len(self.frame_names)):
            CTkMessagebox(title="Error", message=f"Frame Index must be between 0 and {len(self.frame_names) - 1}.", icon="cancel")
            return

        image_path = os.path.join(self.video_dir, self.frame_names[frame_idx])
        image = Image.open(image_path)
        PromptWindow(self, frame_idx, image)

    def reset_predictions(self):
        if self.backend and self.backend.inference_state:
            try:
                self.backend.reset()
                self.prompts_added = False
                CTkMessagebox(title="Success", message="Predictions have been reset.", icon="check")
            except Exception as e:
                CTkMessagebox(title="Error", message=f"Failed to reset predictions:\n{e}", icon="cancel")
        else:
            CTkMessagebox(title="Warning", message="Inference state not initialized.", icon="warning")

    def propagate_masks(self):
        if not self.backend or not self.backend.inference_state:
            CTkMessagebox(title="Error", message="Please initialize inference first.", icon="cancel")
            return
        if not self.prompts_added:
            CTkMessagebox(title="Error", message="Please add at least one prompt before propagating.", icon="cancel")
            return

        self.prop_button.configure(state=ctk.DISABLED)
        self.prop_progress.pack(pady=(0, 5), padx=10, fill="x")
        self.prop_progress_label.configure(text="Starting propagation...")
        self.prop_progress.set(0)

        threading.Thread(target=self._propagate_worker, daemon=True).start()

    def _propagate_worker(self):
        total_frames = len(self.frame_names)
        def ui_progress(cur, total):
            self.root.after(0, self._update_prop_progress, cur, total)

        try:
            self.backend.propagate(self.output_dir, self.colors_custom, progress_callback=ui_progress)
            err = None
        except Exception as e:
            err = str(e)

        def finalize():
            if err:
                self.prop_progress_label.configure(text="Propagation Failed.")
                CTkMessagebox(title="Propagation Error", message=err, icon="cancel")
            else:
                self.prop_progress_label.configure(text="Propagation Complete!")
                CTkMessagebox(title="Success", message="Mask propagation complete.", icon="check")
            self.prop_button.configure(state=ctk.NORMAL)
            self.prop_progress.pack_forget()


        self.root.after(0, finalize)

    def _update_prop_progress(self, current, total):
        self.prop_progress.set(current / total)
        self.prop_progress_label.configure(text=f"Processed frame {current}/{total}")


if __name__ == "__main__":
    ctk.set_appearance_mode("dark") 
    ctk.set_default_color_theme("dark-blue")
    
    root = ctk.CTk()
    app = MaskApp(root)
    root.mainloop()