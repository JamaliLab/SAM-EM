# mask_component.py
import os, json, gc
from typing import Callable, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as e:
    raise RuntimeError(
        "The 'sam2' library is not installed. Install with:\n"
        "pip install 'git+https://github.com/facebookresearch/sam2.git'"
    ) from e


class MaskComponent:
    """
    Headless backend for SAM video segmentation.
    - Loads model
    - Initializes video state
    - Adds prompts (box/points)
    - Propagates masks and saves outputs (npz, masks, combined, index.jsonl)
    """

    def __init__(self, model_cfg: str, model_checkpoint: str, device: Optional[torch.device] = None):
        if not os.path.exists(model_cfg):
            raise FileNotFoundError(f"Config not found: {model_cfg}")
        if not os.path.exists(model_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {model_checkpoint}")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = build_sam2_video_predictor(model_cfg, model_checkpoint, device=self.device)

        # runtime
        self.inference_state = None
        self.video_dir = None
        self.frame_names: List[str] = []

        # output folders (set on propagate)
        self.npz_folder = None
        self.masks_folder = None
        self.combined_folder = None
        self.index_path = None

    # ---------- Utility ----------

    @staticmethod
    def _sort_key(name: str):
        stem, _ = os.path.splitext(name)
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        h = hex_color.lstrip('#')
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    # ---------- Public API ----------

    def init_video(self, video_dir: str) -> List[str]:
        if not os.path.isdir(video_dir):
            raise NotADirectoryError(f"Video directory not found: {video_dir}")
        self.video_dir = video_dir
        self.inference_state = self.predictor.init_state(video_path=video_dir)
        self.frame_names = sorted(
            [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]],
            key=self._sort_key
        )
        return self.frame_names

    def reset(self) -> None:
        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)

    def add_box(self, frame_idx: int, particle_id: int, box_xyxy: List[float]) -> Tuple[List[int], List[np.ndarray]]:
        """
        Returns (obj_ids, masks_bool_list) for visualization.
        """
        if self.inference_state is None:
            raise RuntimeError("Inference not initialized. Call init_video first.")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            self.inference_state, frame_idx, particle_id, box=np.array(box_xyxy, dtype=np.float32)
        )
        obj_ids = [int(x) for x in out_obj_ids]
        masks = [((out_mask_logits[i] > 0.0).squeeze().detach().cpu().numpy().astype(bool)) for i in range(len(obj_ids))]
        return obj_ids, masks

    def add_points(self, frame_idx: int, particle_id: int, points: np.ndarray, labels: np.ndarray) -> Tuple[List[int], List[np.ndarray]]:
        """
        points: (N,2) float32; labels: (N,) float32
        Returns (obj_ids, masks_bool_list).
        """
        if self.inference_state is None:
            raise RuntimeError("Inference not initialized. Call init_video first.")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            self.inference_state, frame_idx, particle_id, points=points.astype(np.float32), labels=labels.astype(np.float32)
        )
        obj_ids = [int(x) for x in out_obj_ids]
        masks = [((out_mask_logits[i] > 0.0).squeeze().detach().cpu().numpy().astype(bool)) for i in range(len(obj_ids))]
        return obj_ids, masks

    def propagate(
        self,
        output_dir: str,
        colors_custom: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Runs full propagation, saving:
         - npz/######.npz (obj_ids, masks)
         - masks/######.png (color mask)
         - combined_results/######.png ([orig|mask])
         - index.jsonl (frame_idx, obj_ids, npz relative path)
        """
        if self.inference_state is None or self.video_dir is None:
            raise RuntimeError("Inference not initialized. Call init_video first.")
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(f"Output directory not found: {output_dir}")

        # Prepare output structure
        self.npz_folder = os.path.join(output_dir, 'npz')
        self.masks_folder = os.path.join(output_dir, 'masks')
        self.combined_folder = os.path.join(output_dir, 'combined_results')
        os.makedirs(self.npz_folder, exist_ok=True)
        os.makedirs(self.masks_folder, exist_ok=True)
        os.makedirs(self.combined_folder, exist_ok=True)
        self.index_path = os.path.join(output_dir, "index.jsonl")

        OVERWRITE = True
        total_frames = len(self.frame_names)

        with open(self.index_path, "a", encoding="utf-8") as index_f:
            for i, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(
                self.predictor.propagate_in_video(self.inference_state)
            ):
                frame_idx = int(out_frame_idx)
                obj_ids = [int(x) for x in out_obj_ids]

                masks_bool = [
                    (out_mask_logits[j] > 0.0).squeeze().detach().cpu().numpy().astype(np.bool_)
                    for j in range(len(obj_ids))
                ]

                # Save npz
                npz_path = os.path.join(self.npz_folder, f"{frame_idx:06d}.npz")
                if OVERWRITE or not os.path.exists(npz_path):
                    np.savez_compressed(
                        npz_path,
                        obj_ids=np.array(obj_ids, dtype=np.int32),
                        masks=np.stack(masks_bool, axis=0).astype(np.uint8)
                    )
                    index_rec = {
                        "frame_idx": frame_idx,
                        "obj_ids": obj_ids,
                        "npz": os.path.relpath(npz_path, output_dir)
                    }
                    index_f.write(json.dumps(index_rec) + "\n")
                    index_f.flush()

                # Save images
                self._save_frame_images(frame_idx, dict(zip(obj_ids, masks_bool)), colors_custom)

                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, total_frames)

                # free memory
                del out_mask_logits, masks_bool
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---------- Internals ----------

    def _save_frame_images(self, frame_idx: int, segments: dict, colors_custom: List[str]) -> None:
        """
        segments: {obj_id: bool_mask}
        Saves:
          - masks/######.png (colorized mask)
          - combined_results/######.png ([orig | mask])
        """
        image_path = os.path.join(self.video_dir, self.frame_names[frame_idx])
        with Image.open(image_path) as orig:
            orig = orig.convert("RGB")
            width, height = orig.size

            mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            num_colors = len(colors_custom)

            for obj_id, mask in segments.items():
                m = np.squeeze(mask).astype(bool)
                r, g, b = self._hex_to_rgb(colors_custom[int(obj_id) % num_colors])
                mask_rgb[m] = (r, g, b)

            mask_img = Image.fromarray(mask_rgb, mode="RGB")
            mask_img.save(os.path.join(self.masks_folder, f"{frame_idx:06d}.png"), optimize=True)

            combined = Image.new("RGB", (width * 2, height))
            combined.paste(orig, (0, 0))
            combined.paste(mask_img, (width, 0))
            combined.save(os.path.join(self.combined_folder, f"{frame_idx:06d}.png"), optimize=True)
