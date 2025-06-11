import rp
import numpy as np
import pandas as pd
from icecream import ic
import torch
from torch.utils.data import Dataset

rp.git_import("remove_watermark")  # Installs https://github.com/RyannDaGreat/remove_watermark

from rp.git.remove_watermark.remove_watermark_envato import remove_watermark

envato_root = "/efs/users/ryan.burgert/public/datasets/envato"
envato_csv_path = rp.path_join(envato_root, "captioned_envato_3869336.csv")
envato_video_root = "/fsx_scanline/from_eyeline/ning_video_genai/datasets/ryan/envato/videos"

# Download the CSV to the local /tmp for faster load times than over /efs
local_envato_csv_path = rp.download_to_cache(envato_csv_path)

if "envato_table" not in vars():
    print("Reading",local_envato_csv_path)
    envato_table = rp.load_csv(local_envato_csv_path, show_progress=True, mode='normal')
    print("Dropna",local_envato_csv_path)
    envato_table = envato_table.dropna()

    captions = envato_table.caption
    video_paths = [rp.path_join(envato_video_root, x) for x in rp.eta(envato_table.videos, 'Path Join')]

    num_videos = len(video_paths)

ic(envato_root, envato_csv_path, envato_video_root, len(envato_table), list(envato_table))


def get_sample(index, start_frame=0, num_frames=49):
    caption = captions[index]
    video_path = video_paths[index]
    video = rp.load_video(video_path, start_frame=start_frame, length=num_frames)
    video = remove_watermark(video)

    # Optional - choose if you want to resize the video
    # video = rp.resize_images(video,size=(480,720))

    # The output format
    assert video.dtype == np.uint8  # Between 0 and 255
    assert video.ndim == 4, video.ndim  # THW3
    assert video.shape[3] == 3  # RGB
    assert isinstance(caption, str)

    return video, caption


class EnvatoDataset(Dataset):
    def __init__(self, start_frame=0, num_frames=49):
        self.start_frame = start_frame
        self.num_frames = num_frames

    def __len__(self):
        return num_videos

    def __getitem__(self, index):
        video, caption = get_sample(index, self.start_frame, self.num_frames)
        video = video.astype(np.float32) / 255

        # Output form
        assert video.ndim == 4  # THWC form
        assert video.shape[3] == 3  # RGB
        assert video.max() <= 1 and video.min() >= 0  # Floating values between 0 and 1

        return video, caption
