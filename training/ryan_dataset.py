"""

    ORIGINAL DATASET FORMAT:
    
        >>> d = cogvideox_image_to_video_lora.VideoDatasetWithResizing(
        ...      data_root      = 'datasets/video-dataset-disney',
        ...      dataset_file   = None,
        ...      caption_column = 'prompt.txt',
        ...      video_column   = 'videos.txt',
        ...      max_num_frames = 49,
        ...      id_token       = 'BW_STYLE',
        ...      height_buckets = [480],
        ...      width_buckets  = [720],
        ...      frame_buckets  = [49],
        ...      load_tensors   = False,
        ...      random_flip    = None,
        ...      image_to_video = True,
        ... )
        ... sample = d[1]
        >>> print(list(sample))             --> ['prompt', 'image', 'video', 'video_metadata']
        >>> print(sample['prompt'])         --> BW_STYLEA black and white animated sequence on a ship's deck features...
        >>> print(sample['image'].shape)    --> torch.Size([1, 3, 480, 720])
        >>> print(sample['image'].min())    --> tensor(-1.)
        >>> print(sample['image'].max())    --> tensor(0.3521)
        >>> print(sample['video'].shape)    --> torch.Size([49, 3, 480, 720])
        >>> print(sample['video_metadata']) --> {'num_frames': 49, 'height': 480, 'width': 720}
        >>> print((sample['image']==sample['video'][0]).all()) --> True

"""

import torch


import rp
rp.git_import("CommonSource")
import einops
import sys

sys.path.append("/root/CleanCode/Github/AnimateDiff_Ning")
import animatediff.data.dataset as ds

def get_sample_helper(
        index,
        debug=False,
        post_noise_alpha=[0,1],
        delegator_address='100.113.78.238',
    ):
    rp.sleep(rp.random_int(1)) #Space them out to prevent errors? Idk....connection reset bs...maybe the webevaluator delegation server can be overloaded by too many requests at a time and just hangs up? I've never tested that??

    if 1 or debug:
        print(f'CALLED get_sample({index})', flush=True)

    index = None #Choose a completely random sample from the delegator! I don't care about epoch perfection

    sample = ds.get_sample_from_delegator(
        index,

        delegator_address=delegator_address, #videotrainer80gb
        
        sample_n_frames=49,
        sample_size=(480, 720),

        # #UNCOMMENT FOR HIGH QUALITY NOISE WARPING - MUST RUN ON 40GB SERVERS
        # S=8,
        # F=7,

        #CHEAP - can run two per-40GB gpu
        S=8,
        F=5,

        # #DIRT CHEAP - for testing only
        # S=8,
        # F=3,

        noise_channels=16,

        # post_noise_alpha = rp.random_float(),
        # post_noise_alpha = 0,
        post_noise_alpha = post_noise_alpha, #Tells the dataset to choose a random number between 0 and 1
        
        delegator_timeout=None,
        csv_path = '/fsx_scanline/from_eyeline/ning_video_genai/datasets/ryan/webvid/webvid_gpt4v_caption_2065605_clean.csv',
    )
    assert set(sample) <= set('text noise pixel_values'.split())

    #Make sample/noise shapes compatible with CogVidX
    sample.noise = einops.rearrange(sample.noise, 'T H W C -> T C H W')
    sample.noise = torch.Tensor(sample.noise)

    #Rename variables for CogVid's codebase
    output = rp.as_easydict(
        instance_prompt = sample.text,
        instance_video = sample.pixel_values,
        instance_noise = sample.noise,
    )

    #Modify this as you code - for clarity.
    assert set('instance_noise instance_video instance_prompt'.split()) == set(output)

    if debug:
        #For debugging
        print(f'get_sample({index}):')
        print(f'    • instance_prompt = {output.instance_prompt}')
        print(f'    • instance_video.shape = {output.instance_video.shape}')
        print(f'    • instance_noise.shape = {output.instance_noise.shape}')
    
    return output


def downsamp_mean(x, l=13):
    return torch.stack([rp.mean(u) for u in rp.split_into_n_sublists(x, l)])

def normalized_noises(noises):
    #Noises is in TCHW form
    return torch.stack([x / x.std(1, keepdim=True) for x in noises])

get_sample_iterator = rp.lazy_par_map(
    rp.squelch_wrap(get_sample_helper),
    [None] * 100000000,
    num_threads=10,
    buffer_limit=10,
)

def get_sample(index=None):
    """
    EXAMPLE:
        >>> sample = get_sample()
        >>> list(sample)
        ans = ['instance_prompt', 'instance_video', 'instance_noise']
        >>> sample.instance_prompt
        ans = The video documents the changing light conditions in the sky, transitioning from bright to a darker, moodier tone. Fluffy clouds against a blue backdrop gradually shift into deeper hues of purple and blue as time progresses, capturing the essence of an early evening sky.
        >>> sample.instance_video.shape
        ans = torch.Size([49, 3, 480, 720])
        >>> sample.instance_video.min(), sample.instance_video.max()
        ans = (tensor(-0.7961), tensor(1.))
        >>> sample.instance_noise.shape
        ans = torch.Size([49, 16, 60, 90])
    """
    while True:
        try:
            output = next(get_sample_iterator)

            if isinstance(output, Exception):
                print("get_sample(): RAISED AN ERROR!")
                raise output
            else:
                print("get_sample(): GOT AN OUTPUT!")
                return output

        except Exception:
            print("get_sample error:")
            rp.print_stack_trace()
            rp.sleep(1)

# next(get_sample_iterator)
# next(get_sample_iterator)
# next(get_sample_iterator)
# get_sample()
# get_sample()
# get_sample()
# quit()


def test_get_sample():
    sample = get_sample(123)
    print(f'set(sample) = {set(sample)}')
    return sample
