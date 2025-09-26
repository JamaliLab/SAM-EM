import os
import re
import numpy as np
import pandas as pd
from skimage.measure import label as cc_label, regionprops
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap

DARK_BG = "#2b2b2b"

class ParticleTrackingUI:
    def __init__(self, parent, app):
        """
        parent : the CTkTab (frame) to build on
        app    : reference to the main MaskApp (so you can call its methods/attributes)
        """
        self.parent = parent
        self.app = app

        # will hold matplotlib Figure objects after analysis
        self.last_figs = {}      # {"tMSD": fig, ...}
        self.csv_path = ""       # set after Generate CSV
        self.plots_window = None # Toplevel when opened

        self._build_ui()
    def _create_colormap(self, color_hex_or_name: str):
        """White → target color gradient colormap."""
        return LinearSegmentedColormap.from_list("custom_cmap", ["white", color_hex_or_name])
    # ---------- UI ----------
    def _build_ui(self):
        # Step 1: Trajectory generation
        step1_frame = ctk.CTkFrame(self.parent)
        step1_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(step1_frame, text="Generate Trajectories from NPZ").grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self.npz_dir_entry = ctk.CTkEntry(step1_frame, width=350)
        self.npz_dir_entry.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(step1_frame, text="Browse NPZ Folder", command=self.browse_npz_dir).grid(row=1, column=1, padx=5, pady=5)

        self.output_dir_entry = ctk.CTkEntry(step1_frame, width=350)
        self.output_dir_entry.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(step1_frame, text="Select Output Folder", command=self.browse_output_dir).grid(row=2, column=1, padx=5, pady=5)

        ctk.CTkButton(step1_frame, text="Generate CSV", command=self.generate_csv).grid(row=3, column=0, columnspan=2, pady=10)

        self.step1_status = ctk.CTkLabel(step1_frame, text="", text_color="gray")
        self.step1_status.grid(row=4, column=0, columnspan=2, pady=5)

        # Step 2: Motion analysis
        step2_frame = ctk.CTkFrame(self.parent)
        step2_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(step2_frame, text="Motion Analysis from CSV").grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self.csv_input_entry = ctk.CTkEntry(step2_frame, width=350)
        self.csv_input_entry.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(step2_frame, text="Browse CSV", command=self.browse_input_csv).grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkButton(step2_frame, text="Run Motion Analysis", command=self.run_analysis).grid(row=2, column=0, pady=10, padx=10, sticky="w")
        self.last_figs = {}
        self.view_btn = ctk.CTkButton(step2_frame, text="View Graphs", command=self.open_plots_window, state=ctk.DISABLED)
        self.view_btn.grid(row=2, column=1, pady=10, padx=5, sticky="e")

        self.step2_status = ctk.CTkLabel(step2_frame, text="", text_color="gray")
        self.step2_status.grid(row=3, column=0, columnspan=2, pady=5)

    # ---------- File pickers ----------
    def browse_npz_dir(self):
        folder = filedialog.askdirectory(title="Select Folder Containing NPZ Files")
        if folder:
            self.npz_dir_entry.delete(0, ctk.END)
            self.npz_dir_entry.insert(0, folder)

    def browse_output_dir(self):
        folder = filedialog.askdirectory(title="Select Output Folder for CSV")
        if folder:
            self.output_dir_entry.delete(0, ctk.END)
            self.output_dir_entry.insert(0, folder)

    def browse_input_csv(self):
        file = filedialog.askopenfilename(title="Select Input CSV", filetypes=[("CSV files", "*.csv")])
        if file:
            self.csv_input_entry.delete(0, ctk.END)
            self.csv_input_entry.insert(0, file)

    # ---------- NPZ -> CSV ----------
    def _frame_from_fname(self, fname: str) -> int:
        m = re.search(r'(\d+)', fname)
        return int(m.group(1)) if m else 0

    def _extract_masks_from_npz(self, npz_path: str):
        """Return (frame_idx, [(particle_id, mask2d_bool), ...])."""
        with np.load(npz_path, allow_pickle=True) as data:
            keys = list(data.keys())

            # 1) optional frame index
            frame_idx = None
            for k in ("frame", "frame_idx", "frame_index"):
                if k in data:
                    try:
                        frame_idx = int(np.array(data[k]).item())
                        break
                    except Exception:
                        pass
            if frame_idx is None:
                frame_idx = self._frame_from_fname(os.path.basename(npz_path))

            # 2) explicit label map
            if "labels" in data and data["labels"].ndim == 2:
                lbl = np.asarray(data["labels"])
                ids = [int(v) for v in np.unique(lbl) if v != 0]
                masks = [(pid, (lbl == pid)) for pid in ids]
                return frame_idx, masks

            # 3) stack of masks
            stack_key = None
            for k in ("masks", "mask_stack", "instance_masks", "arr_0", "mask"):
                if k in data:
                    arr = np.asarray(data[k])
                    if arr.ndim in (2, 3):
                        stack_key = k
                        break

            if stack_key is not None:
                arr = np.asarray(data[stack_key])

                if arr.ndim == 2:
                    lbl = cc_label(arr > 0)
                    ids = [int(v) for v in np.unique(lbl) if v != 0]
                    masks = [(pid, (lbl == pid)) for pid in ids]
                    return frame_idx, masks

                if arr.ndim == 3:
                    # guess N,H,W vs H,W,N
                    if arr.shape[0] <= 64 and arr.shape[0] != arr.shape[1]:
                        stack = arr
                    elif arr.shape[-1] <= 64:
                        stack = np.transpose(arr, (2, 0, 1))
                    else:
                        stack = arr

                    obj_ids = None
                    for k in ("obj_ids", "ids", "particle_ids"):
                        if k in data:
                            try:
                                obj_ids = [int(x) for x in np.array(data[k]).ravel().tolist()]
                            except Exception:
                                pass
                            break

                    masks = []
                    for i in range(stack.shape[0]):
                        pid = obj_ids[i] if (obj_ids and i < len(obj_ids)) else (i + 1)
                        masks.append((pid, stack[i] > 0))
                    return frame_idx, masks

            # 4) fallback: first 2D thing as mask
            for k in keys:
                arr = np.asarray(data[k])
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    lbl = cc_label(arr > 0)
                    ids = [int(v) for v in np.unique(lbl) if v != 0]
                    masks = [(pid, (lbl == pid)) for pid in ids]
                    return frame_idx, masks

            return frame_idx, []

    def generate_csv(self):
        npz_dir = self.npz_dir_entry.get()
        out_dir = self.output_dir_entry.get()
        if not npz_dir or not out_dir:
            CTkMessagebox(title="Error", message="Please select NPZ folder and output folder.", icon="cancel")
            return

        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "trajectories.csv")

        rows, skipped = [], []
        for fname in sorted(os.listdir(npz_dir)):
            if not fname.lower().endswith(".npz"):
                continue
            fpath = os.path.join(npz_dir, fname)
            try:
                frame_idx, masks = self._extract_masks_from_npz(fpath)
                if not masks:
                    skipped.append(f"{fname} (no masks found)")
                    continue

                for pid, m in masks:
                    lbl = cc_label(m)
                    for comp in regionprops(lbl):
                        cy, cx = comp.centroid
                        angle_deg = float(np.degrees(getattr(comp, "orientation", 0.0)))
                        rows.append([frame_idx, int(pid), float(cx), float(cy), angle_deg])
            except Exception as e:
                skipped.append(f"{fname} ({e})")

        if not rows:
            msg = "No trajectories were written."
            if skipped:
                msg += "\n\nSkipped files:\n" + "\n".join(skipped[:20])
            CTkMessagebox(title="No Data", message=msg, icon="warning")
            return

        df = pd.DataFrame(rows, columns=["frame", "particle_id", "x", "y", "angle"])
        df.sort_values(["particle_id", "frame"], inplace=True)
        df.to_csv(csv_path, index=False)

        self.csv_path = csv_path
        note = f"CSV saved to output directory!"
        if skipped:
            note += f"\nSkipped {len(skipped)} file(s)."
        self.step1_status.configure(text=note, text_color="green")

    # ---------- Analysis + Plots ----------
    def _apply_dark(self, ax):
        ax.set_facecolor(DARK_BG)
        ax.figure.patch.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#bbbbbb")

    def _style_colorbar(self, cbar):
        cbar.ax.tick_params(colors="white")
        cbar.outline.set_edgecolor("white")
        if cbar.ax.yaxis.label:
            cbar.ax.yaxis.label.set_color("white")

    def run_analysis(self):
        # pick CSV: entry first, otherwise last generated
        self.last_figs = {}
        csv_file = self.csv_input_entry.get() or getattr(self, "csv_path", "")
        if not csv_file or not os.path.isfile(csv_file):
            CTkMessagebox(title="Error", message="Please select a valid CSV file for analysis.", icon="cancel")
            return

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            CTkMessagebox(title="Error", message=f"Failed to read CSV:\n{e}", icon="cancel")
            return

        # We expect these columns (your CSV writer already uses them)
        needed = {"frame", "particle_id", "x", "y"}
        if not needed.issubset(df.columns):
            CTkMessagebox(title="Error",
                        message=f"CSV must contain: {', '.join(sorted(needed))}",
                        icon="cancel")
            return

        # Clean & sort
        df = df[list(needed)].dropna()
        for c in list(needed):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        if df.empty:
            CTkMessagebox(title="Error", message="No usable rows after cleaning.", icon="cancel")
            return
        df.sort_values(["particle_id", "frame"], inplace=True)

        # Timing: fps from entry or default 1.0
        try:
            fps = float(getattr(self, "fps_entry", None).get()) if hasattr(self, "fps_entry") else 1.0
            fps = fps if fps > 0 else 1.0
        except Exception:
            fps = 1.0
        dt = 1.0 / fps

        # Colors: use app.colors_custom if available, else a default palette
        base_colors = getattr(self.app, "colors_custom", [
            "#E69F00", "#D55E00", "#56B4E9", "#009E73", "#0072B2",
            "#CC79A7", "#F0E442", "#B3D100", "#9E1B32"
        ])
        pids = list(df["particle_id"].drop_duplicates())
        # map pid → solid color and pid → white→color cmap
        pid_to_color = {pid: base_colors[i % len(base_colors)] for i, pid in enumerate(pids)}
        pid_to_cmap  = {pid: self._create_colormap(pid_to_color[pid]) for pid in pids}

        # ---------- helper styling ----------
        def apply_dark(ax):
            ax.set_facecolor("#2b2b2b")
            ax.figure.patch.set_facecolor("#2b2b2b")
            ax.tick_params(colors="white", labelsize=20)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#bbbbbb")

        # ---------- 1) Distribution of Displacements (per particle) ----------
        # your snippet: r = sqrt(x^2 + y^2), disp_r = diff(r), plot per-object colored hist
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        apply_dark(ax_dist)

        all_abs = []
        for pid in pids:
            g = df[df["particle_id"] == pid]
            x = g["x"].to_numpy()
            y = g["y"].to_numpy()
            if len(x) < 2:
                continue
            r = np.sqrt(x**2 + y**2)
            disp_r = np.diff(r)  # Δr between frames
            all_abs.append(np.abs(disp_r))

            # emulate your plot_hist(.., color=cmap_custom(pid))
            ax_dist.hist(
                disp_r, bins=40, density=True, alpha=0.45,  # translucent
                color=pid_to_color[pid], edgecolor=None
            )

        # axis labels & look
        ax_dist.set_xlabel("Δr", fontsize=20)
        ax_dist.set_ylabel("Normalized PDF", fontsize=20)
        ax_dist.set_title("Distribution of Displacements", fontsize=20)

        # set symmetric x-limits like your example ([-50, 50]) but adapt to data
        if all_abs:
            max_abs = float(np.percentile(np.concatenate(all_abs), 99.5))  # robust
            if max_abs == 0:
                max_abs = 1.0
            ax_dist.set_xlim([-max_abs, max_abs])

        # ---------- 2) tMSD (per particle, log–log) ----------
        # your example builds time in seconds and scatters MSD vs tau
        def msd_xy(x, y):
            coords = np.stack([x, y], axis=1)
            N = len(coords)
            taus = np.arange(1, N)  # 1..N-1
            msd = np.empty_like(taus, dtype=float)
            for i, tau in enumerate(taus):
                d = coords[tau:] - coords[:-tau]
                msd[i] = np.mean(np.einsum("ij,ij->i", d, d))
            return taus, msd

        fig_msd, ax_msd = plt.subplots(figsize=(6, 4))
        apply_dark(ax_msd)

        for pid in pids:
            g = df[df["particle_id"] == pid]
            x = g["x"].to_numpy()
            y = g["y"].to_numpy()
            if len(x) < 3:
                continue
            taus, msd_vals = msd_xy(x, y)
            t_sec = taus * dt
            ax_msd.scatter(t_sec, msd_vals, s=12, lw=0, color=pid_to_color[pid], label=f"{pid}")

        ax_msd.set_xscale("log"); ax_msd.set_yscale("log")
        ax_msd.set_title("MSD", fontsize=20)
        ax_msd.set_xlabel(r"$\tau$ (s)", fontsize=20)
        ax_msd.set_ylabel(r"$\overline{\delta r^2(\tau)}$ (nm$^2$)", fontsize=20)
        if len(pids) <= 15:  # avoid huge legends
            leg = ax_msd.legend(title="particle", fontsize=14)
            for text in leg.get_texts():
                text.set_color("white")
            leg.get_title().set_color("white")

        # ---------- 3) Trajectories (white→color gradient by time, per particle) ----------
        fig_traj, ax_traj = plt.subplots(figsize=(6, 5))
        apply_dark(ax_traj)

        last_scat = None
        for pid in pids:
            g = df[df["particle_id"] == pid]
            x = g["x"].to_numpy(); y = g["y"].to_numpy()
            if len(x) < 2:
                continue
            t = np.linspace(0, (len(g)-1)*dt, len(g))
            cmap_i = pid_to_cmap[pid]

            # scatter points colored by time
            scat = ax_traj.scatter(x, y, c=t, s=6, zorder=1, cmap=cmap_i)
            last_scat = scat  # for colorbar

            # color line segments with same gradient
            colors = scat.to_rgba(t[:-1])
            for i in range(len(x)-1):
                ax_traj.plot([x[i], x[i+1]], [y[i], y[i+1]], color=colors[i], lw=2, zorder=0)

        if last_scat is not None:
            cbar = fig_traj.colorbar(last_scat, ax=ax_traj)
            cbar.set_label('t(s)', size=20, rotation=0, labelpad=15)
            cbar.ax.tick_params(colors="white", labelsize=16)
            cbar.outline.set_edgecolor("white")

        ax_traj.set_xlabel('x', fontsize=20)
        ax_traj.set_ylabel('y', fontsize=20)
        ax_traj.set_title("Trajectories", fontsize=20)
        ax_traj.set_aspect('equal', 'datalim')
        ax_traj.invert_yaxis()  # match your snippet

        # If you know physical field size, set like (0,width_nm) and (height_nm, 0).
        # Otherwise, keep data-driven limits (already set by plotting).

        # ---------- 4) Autocorrelation (radial velocity ACF, first ~50 lags) ----------
        def norm_autocorr(x):
            """Normalized autocorrelation of a 1D series (x - mean), unbiased denominator."""
            x = np.asarray(x)
            x = x - np.mean(x)
            N = len(x)
            if N < 3:
                return np.array([0.0])
            denom = np.sum(x*x)
            ac = np.correlate(x, x, mode="full")[N-1:] / (denom + 1e-12)
            return ac

        fig_acf, ax_acf = plt.subplots(figsize=(6, 4))
        apply_dark(ax_acf)

        max_lag = 50
        for pid in pids:
            g = df[df["particle_id"] == pid]
            x = g["x"].to_numpy(); y = g["y"].to_numpy()
            if len(x) < 3:
                continue
            r = np.sqrt(x**2 + y**2)
            v_r = np.diff(r) / dt  # radial velocity
            z = norm_autocorr(v_r)
            L = min(max_lag, len(z)-1)
            ax_acf.plot(np.arange(0, L), z[:L], color=pid_to_color[pid], lw=2)

        ax_acf.set_title("Autocorrelation", fontsize=20)
        ax_acf.set_xlabel(r'lag time $\tau$ (s)', fontsize=20)
        ax_acf.set_ylabel(r'$C_v(\tau)$', fontsize=20)

        # ---------- store figures & enable "View Graphs" ----------
        self.last_figs = {
            "Displacements": fig_dist,
            "tMSD": fig_msd,
            "Trajectories": fig_traj,
            "Autocorrelation": fig_acf,
        }
        self.view_btn.configure(state=ctk.NORMAL)
        self.step2_status.configure(text=f"Analysis complete for {os.path.basename(csv_file)}", text_color="green")

    # ---------- Plots window (separate) ----------
    def open_plots_window(self):
        if not self.last_figs:
            CTkMessagebox(title="No Plots", message="Run Motion Analysis first.", icon="warning")
            return

        # Close an existing window if it's still around
        try:
            if self.plots_window and self.plots_window.winfo_exists():
                self.plots_window.destroy()
        except Exception:
            pass

        win = ctk.CTkToplevel(self.parent)
        win.title("Motion Analysis Graphs")
        win.geometry("1000x750")
        self.plots_window = win

        # Toolbar (Save buttons)
        toolbar = ctk.CTkFrame(win, fg_color="transparent")
        toolbar.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(toolbar, text="Save Current…", command=self._save_current_plot).pack(side="left", padx=(0, 6))
        ctk.CTkButton(toolbar, text="Save All…", command=self._save_all_plots).pack(side="left")

        # Tabs with canvases
        self.plots_tabview = ctk.CTkTabview(win)
        self.plots_tabview.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Keep canvases alive
        self._plot_canvases = {}

        for name, fig in self.last_figs.items():
            tab = self.plots_tabview.add(name)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._plot_canvases[name] = canvas

        # Clean up figures when window closes
        def _on_close():
            try:
                for fig in self.last_figs.values():
                    plt.close(fig)
            except Exception:
                pass
            self._plot_canvases = {}
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

    # ---------- Saving ----------
    def _save_current_plot(self):
        if not self.last_figs:
            CTkMessagebox(title="No Plots", message="No plots to save.", icon="warning")
            return
        # Current tab name
        current = self.plots_tabview.get()
        fig = self.last_figs.get(current)
        if fig is None:
            CTkMessagebox(title="Error", message="No current plot selected.", icon="cancel")
            return

        path = filedialog.asksaveasfilename(
            title=f"Save {current} as...",
            defaultextension=".png",
            initialfile=f"{current}.png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")]
        )
        if not path:
            return
        try:
            fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
            CTkMessagebox(title="Saved", message=f"Saved: {os.path.basename(path)}", icon="check")
        except Exception as e:
            CTkMessagebox(title="Error", message=f"Failed to save:\n{e}", icon="cancel")

    def _save_all_plots(self):
        if not self.last_figs:
            CTkMessagebox(title="No Plots", message="No plots to save.", icon="warning")
            return
        folder = filedialog.askdirectory(title="Select Folder to Save All Plots")
        if not folder:
            return
        errors = []
        for name, fig in self.last_figs.items():
            try:
                out = os.path.join(folder, f"{name}.png")
                fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
            except Exception as e:
                errors.append(f"{name}: {e}")
        if errors:
            CTkMessagebox(title="Partial Save", message="Some plots failed:\n" + "\n".join(errors), icon="warning")
        else:
            CTkMessagebox(title="Saved", message=f"All plots saved to:\n{folder}", icon="check")
