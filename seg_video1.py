import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


def merget_video(input_files, out_path):
    dir_name = os.path.dirname(out_path)
    os.makedirs(dir_name, exist_ok=True)
    os.system(f"ffmpeg -framerate 25 -i {input_files} -c:v libx264 -pix_fmt yuv420p {out_path} -y")
    print(f"see {out_path}")


def seg_video(video_path, output_dir, points, labels, merge_only=False):
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    out_mask_dir = f"{output_dir}/sam2_mask"
    os.makedirs(out_mask_dir, exist_ok=True)
    if merge_only:
        merget_video(f"{out_mask_dir}/%6d.jpg", f"{output_dir}/sam2.mp4")
        return

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path)
        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, np.int32)
        # add new prompts and instantly get the output on the same frame
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            first_mask = ((masks[0] > 0.0).squeeze() * 255).cpu().numpy().astype(np.uint8)
            # print(first_mask.shape)
            save_path = f"{out_mask_dir}/{frame_idx:06d}.jpg"
            cv2.imwrite(save_path, first_mask)
            # print(save_path)

    merget_video(f"{out_mask_dir}/%6d.jpg", f"{output_dir}/sam2.mp4")


dataset = [
    ("HSID48", [[361, 411], [308, 116], [372, 618]], [1, 0, 0]),
    ("HSID13", [[231, 296], [223, 415], [220, 49]], [1, 0, 0]),
]
dataset = [
    ("demo/data/gallery/01_dog.mp4", "res", [[432, 249], [323, 299], [300, 397], [262, 345]], [1, 1, 0, 0]),
]
if __name__ == "__main__":
    for data in dataset:
        seg_video(*data, merge_only=True)
