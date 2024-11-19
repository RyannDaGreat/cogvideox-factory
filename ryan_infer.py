from rp import *
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers import CogVideoXVideoToVideoPipeline
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video, load_image
from icecream import ic
import rp.git.CommonSource.noise_warp as nw
import training.ryan_dataset as ryan_dataset

pipe_ids = dict(
    T2V5B="THUDM/CogVideoX-5b",
    T2V2B="THUDM/CogVideoX-2b",
    I2V5B="THUDM/CogVideoX-5b-I2V",
)
# From a bird's-eye view, a serene scene unfolds: a herd of deer gracefully navigates shallow, warm-hued waters, their silhouettes stark against the earthy tones. The deer, spread across the frame, cast elongated, well-defined shadows that accentuate their antlers, creating a mesmerizing play of light and dark. This aerial perspective captures the tranquil essence of the setting, emphasizing the harmonious contrast between the deer and their mirror-like reflections on the water's surface. The composition exudes a peaceful stillness, yet the subtle movement suggested by the shadows adds a dynamic layer to the natural beauty and symmetry of the moment.
lora_paths = dict(
    T2V5B_RDeg_i9800         = '/root/CleanCode/Github/CogVideo/finetune/cogvideox5b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate/checkpoint-9800/saved_weights_copy/pytorch_lora_weights.safetensors',
    T2V5B_0Deg_L512_ND_i1200 = '/root/CleanCode/Github/CogVideo/finetune/cogvideox5b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate-LORA512-0Degrad/checkpoint-1200/saved_weights_copy/pytorch_lora_weights.safetensors',
    T2V2B_RDeg_i30000        = '/root/CleanCode/Github/CogVideo/finetune/cogvideox2b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate/checkpoint-30000/saved_weights_copy/pytorch_lora_weights.safetensors',
    T2V5B_RDeg_L2048_i4800   = '/root/CleanCode/Github/CogVideo/finetune/cogvideox5b-lora-noisewarp-Oct23-LORA2048-RandDegrad-BlendNoiseWithoutNorm/checkpoint-4800/saved_weights_copy/pytorch_lora_weights.safetensors',
    # ...
    I2V5B_i2v_webvid_i2600   = '/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v__degrad=0,1__downtemp=nearest__lr=1e-4__2024-10-25T14-52-57-0400/checkpoint-2600/pytorch_lora_weights.safetensors', #Oct26, 3:45AM
    I2V5B_i2v_webvid_i3200   = '/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v__degrad=0,1__downtemp=nearest__lr=1e-4__2024-10-25T14-52-57-0400/checkpoint-3200/pytorch_lora_weights.safetensors', #Oct26, 6:50AM

    I2V5B_resum_blendnorm_0degrad_i5000_webvid  = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v__ZeroDegrad__resume=CHECKPOINT_I2V5B_i2v_webvid_i3200__degrad=0__downtemp=blend_norm__lr=1e-4__2024-10-27T04-42-17-0400/checkpoint-5000/pytorch_lora_weights.safetensors",
    I2V5B_resum_blendnorm_0degrad_i7600_webvid  = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v__ZeroDegrad__resume=CHECKPOINT_I2V5B_i2v_webvid_i3200__degrad=0__downtemp=blend_norm__lr=1e-4__2024-10-27T04-42-17-0400/checkpoint-7600/pytorch_lora_weights.safetensors",
    I2V5B_resum_blendnorm_0degrad_i13600_webvid = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v__ZeroDegrad__resume=CHECKPOINT_I2V5B_i2v_webvid_i3200__degrad=0__downtemp=blend_norm__lr=1e-4__2024-10-27T04-42-17-0400/checkpoint-13600/pytorch_lora_weights.safetensors",
    I2V5B_resum_blendnorm_i5400_webvid          = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v_CHECKPOINT_I2V5B_i2v_webvid_i3200__degrad=0,1__downtemp=blend_norm__lr=1e-4__2024-10-27T04-18-13-0400/checkpoint-5200/pytorch_lora_weights.safetensors",
    I2V5B_resum_blendnorm_i6400_webvid          = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v_CHECKPOINT_I2V5B_i2v_webvid_i3200__degrad=0,1__downtemp=blend_norm__lr=1e-4__2024-10-27T04-18-13-0400/checkpoint-6400/pytorch_lora_weights.safetensors",
    I2V5B_resum_blendnorm_i7600_webvid          = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v_CHECKPOINT_I2V5B_i2v_webvid_i3200__degrad=0,1__downtemp=blend_norm__lr=1e-4__2024-10-27T04-18-13-0400/checkpoint-7600/pytorch_lora_weights.safetensors",
    I2V5B_resum_blendnorm_i13400_webvid         = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v_CHECKPOINT_I2V5B_i2v_webvid_i3200__degrad=0,1__downtemp=blend_norm__lr=1e-4__2024-10-27T04-18-13-0400/checkpoint-13400/pytorch_lora_weights.safetensors",
    I2V5B_resum_blendnorm_i22600_webvid         = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v__EnvatoFromWebvid__resume=CHECKPOINT_I2V5B_i2v_webvid_i13400__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={2048}__2024-10-30T10-58-22-0400/checkpoint-22600/pytorch_lora_weights.safetensors",
    I2V5B_resum_blendnorm_i26600_webvid         = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v__EnvatoFromWebvid__resume=CHECKPOINT_I2V5B_i2v_webvid_i13400__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={2048}__2024-10-30T10-58-22-0400/checkpoint-26600/pytorch_lora_weights.safetensors",
    I2V5B_resum_blendnorm_i30000_webvid         = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v__EnvatoFromWebvid__resume=CHECKPOINT_I2V5B_i2v_webvid_i13400__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={2048}__2024-10-30T10-58-22-0400/checkpoint-29800/pytorch_lora_weights.safetensors",
    I2V5B_final_i30000                          = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-EnvatoFromWebvidContinued__resume=CHECKPOINT_I2V5B_resum_blendnorm_i26600__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={2048}__2024-11-03T21-11-57-0500/checkpoint-29800/pytorch_lora_weights.safetensors",
    I2V5B_final_i38800_nearest                  = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-NEAREST_OVERTIME__EnvatoFromWebvidContinued____resume=CHECKPOINT_I2V5B_resum_blendnorm_i26600__degrad=0,1__downtemp=nearest__lr=1e-4__rank={2048}__2024-11-10T09-48-33-0500/checkpoint-38800/pytorch_lora_weights.safetensors",

    T2V5B_blendnorm_i1800_envato         = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvatoFromScratch__resume=__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-03T15-35-06-0500/checkpoint-1800/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i2000_envato         = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvatoFromScratch__resume=__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-03T15-35-06-0500/checkpoint-2000/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i2800_envato         = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvatoFromScratch__resume=__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-03T15-35-06-0500/checkpoint-2800/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i6800_envato         = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvatoFromScratch__resume=__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-03T15-35-06-0500/checkpoint-6800/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i7400_envato         = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvatoFromScratch__resume=__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-03T15-35-06-0500/checkpoint-7400/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i9600_envato         = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvatoFromScratch__resume=__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-03T15-35-06-0500/checkpoint-9600/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i11600_envato        = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvatoFromScratch__resume=__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-03T15-35-06-0500/checkpoint-11600/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i16400_envato        = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvatoFromScratch__resume=__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-03T15-35-06-0500/checkpoint-16400/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i16800_envato        = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvatoFromScratch__resume=__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-03T15-35-06-0500/checkpoint-16800/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i18000_envato        = "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvatoFromScratch__resume=__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-03T15-35-06-0500/checkpoint-18000/pytorch_lora_weights.safetensors",
    
    T2V5B_blendnorm_i11000_envato_nearest= "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvato__ResumeWithNearest____resume=CHECKPOINT_T2V5B_blendnorm_i9400_envato__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-05T16-00-32-0500/checkpoint-11000/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i16400_envato_nearest= "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvato__ResumeWithNearest____resume=CHECKPOINT_T2V5B_blendnorm_i9400_envato__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-05T16-00-32-0500/checkpoint-16400/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i18000_envato_nearest= "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvato__ResumeWithNearest____resume=CHECKPOINT_T2V5B_blendnorm_i11200_envato__degrad=0,1__downtemp=blend_norm__lr=1e-4__rank={3072}__2024-11-06T01-17-49-0500/checkpoint-18000/pytorch_lora_weights.safetensors",
    T2V5B_blendnorm_i25000_envato_nearest= "/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-TextToVideoFromEnvato__ResumeWithNearest____resume=CHECKPOINT_T2V5B_blendnorm_i18000_envato_nearest__degrad=0,1__downtemp=nearest__lr=1e-4__rank={3072}__2024-11-10T09-44-21-0500/checkpoint-25000/pytorch_lora_weights.safetensors",
)
#To get the trained LoRA paths:
#     >>> lora_paths =glob.glob('/root/CleanCode/Github/CogVideo/finetune/*/*/saved_weights_copy/pytorch_lora_weights.safetensors') #For Old Training Codebase (T2V)
#     >>> lora_paths+=glob.glob('/root/CleanCode/Github/cogvideox-factory/outputs/models/*/*/*.safetensors')                        #For New Training Codebase (I2V)
#     >>> def get_lora_name(x): return [y for y in x.split("/") if "lora" in y][0]
#     >>> print(line_join(sorted([max(x, key=by_number) for x in cluster_by_key(lora_paths, get_lora_name)], key=date_created)))
#     ... #OUTPUT:
#     ... # /root/CleanCode/Github/CogVideo/finetune/cogvideox5b-lora-noisewarp-Oct23-LORA2048-RandDegrad-BlendNoiseWithoutNorm/checkpoint-4800/saved_weights_copy/pytorch_lora_weights.safetensors
#     ... # /root/CleanCode/Github/CogVideo/finetune/cogvideox5b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate/checkpoint-9800/saved_weights_copy/pytorch_lora_weights.safetensors
#     ... # /root/CleanCode/Github/CogVideo/finetune/cogvideox2b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate/checkpoint-30000/saved_weights_copy/pytorch_lora_weights.safetensors
#     ... # /root/CleanCode/Github/CogVideo/finetune/cogvideox5b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate-LORA512-0Degrad/checkpoint-1200/saved_weights_copy/pytorch_lora_weights.safetensors
#     ... # /root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v__degrad=0,1__downtemp=nearest__lr=1e-4__2024-10-25T14-52-57-0400/checkpoint-2600/pytorch_lora_weights.safetensors

dtype=torch.bfloat16

#https://medium.com/@ChatGLM/open-sourcing-cogvideox-a-step-towards-revolutionizing-video-generation-28fa4812699d
B, F, C, H, W = 1, 13, 16, 60, 90  # The defaults
num_frames=(F-1)*4+1 #https://miro.medium.com/v2/resize:fit:1400/format:webp/0*zxsAG1xks9pFIsoM
#Possible num_frames: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49
assert num_frames==49

@memoized #Torch never manages to unload it from memory anyway
def get_pipe(pipe_name=None, lora_name=None, device=None):
    assert pipe_name is not None or lora_name is not None

    if pipe_name is None and isinstance(lora_name, str):
        #By convention, we have lora_paths that start with the pipe names - such as 
        fansi_print(f"Getting pipe name from lora_name={lora_name}",'cyan','bold')
        pipe_name = lora_name.split('_')[0]

    is_i2v = "I2V" in pipe_name  # This is a convention I'm using right now
    is_v2v = "V2V" in pipe_name  # This is a convention I'm using right now

    if is_v2v:
        old_pipe_name = pipe_name
        old_lora_name = lora_name
        if pipe_name is not None: pipe_name = pipe_name.replace('V2V','T2V')
        if lora_name is not None: lora_name = lora_name.replace('V2V','T2V')
        rp.fansi_print(f"V2V: {old_pipe_name} --> {pipe_name}   &&&   {old_lora_name} --> {lora_name}",'white','bold italic','red')
    
    pipe_id = pipe_ids[pipe_name]
    print(f"LOADING PIPE WITH device={device} pipe_name={pipe_name} pipe_id={pipe_id} lora_name={lora_name}")
    
    PipeClass = CogVideoXImageToVideoPipeline if is_i2v else CogVideoXPipeline
    if is_v2v:
        PipeClass = CogVideoXVideoToVideoPipeline

    pipe = PipeClass.from_pretrained(pipe_ids[pipe_name], torch_dtype=torch.bfloat16)

    pipe.pipe_name = pipe_name

    if lora_name is not None:
        lora_path = lora_paths[lora_name]
        assert file_exists(lora_path), (lora_name, lora_path)
        print(end="\tLOADING LORA WEIGHTS...",flush=True)
        pipe.load_lora_weights(rp.download_file_to_cache(lora_path))
        print("DONE!")

    if device is None:
        device = select_torch_device()

    if device is not None:
        print("\tUSING PIPE DEVICE", device)
        pipe = pipe.to(device)

    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # Metadata
    pipe.lora_name = lora_name
    pipe.pipe_name = pipe_name
    pipe.is_i2v    = is_i2v
    pipe.is_v2v    = is_v2v
    
    return pipe

@memoized
def load_sample_cartridge(
    sample_path: str = None,
    degradation=0,
    noise_downtemp_interp='nearest',
    image=None,
    prompt=None,
    #SETTINGS:
    num_inference_steps=30,
    guidance_scale=6,
    v2v_strength=.5,
):
    """
    COMPLETELY FROM SAMPLE: Generate with /root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidSampleGenerator.ipynb
    EXAMPLE PATHS:
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/plus_pug.pkl'
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/amuse_chop.pkl'
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/chomp_shop.pkl'
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ahead_job.pkl'
        sample_path = rp.random_element(glob.glob('/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/*.pkl'))
    """

    #These could be args in the future. I can't think of a use case yet though, so I'll keep the signature clean.
    noise=None
    video=None

    if sample_path is None:
        #Choose somethhing
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/amend_shred.pkl' #Driving on a road - lots of zoom
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/clink_grief.pkl' #Camera curving forward to left

    print(end="LOADING "+sample_path+"...")
    sample=rp.file_to_object(sample_path)
    print("DONE!")

    #SAMPLE EXAMPLE:
    #    >>> sample=file_to_object('/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ahead_job.pkl')
    #    >>> list(sample)?s                 -->  ['instance_prompt', 'instance_video', 'instance_noise']
    #    >>> sample.instance_prompt?s       -->  A group of elk, including a dominant bull, is seen grazing and moving through...
    #    >>> sample.instance_noise.shape?s  -->  torch.Size([49, 16,  60,  90])
    #    >>> sample.instance_video.shape?s  -->  torch.Size([49,  3, 480, 720])

    sample_noise  = sample.instance_noise.to(dtype)
    sample_video  = sample.instance_video.to(dtype)
    sample_prompt = sample.instance_prompt

    sample_gif_path = sample_path+'.mp4'
    if not rp.file_exists(sample_gif_path):
        sample_gif_path = sample_path+'.gif' #The older scripts made this. Backwards compatibility.
    if not rp.file_exists(sample_gif_path):
        #Create one!
        #Clientside warped noise does not come with a nice GIF so we make one here and now!
        sample_gif_path = sample_path+'.mp4'

        rp.fansi_print("MAKING SAMPLE PREVIEW VIDEO",'light blue green','underlined')
        preview_sample_video=rp.as_numpy_images(sample_video)/2+.5
        preview_sample_noise=rp.as_numpy_images(sample_noise)[:,:,:,:3]/5+.5
        preview_sample_noise = rp.resize_images(preview_sample_noise, size=8, interp="nearest")
        preview_sample=rp.horizontally_concatenated_videos(preview_sample_video,preview_sample_noise)
        save_video_mp4(preview_sample,sample_gif_path,video_bitrate='max',framerate=12)
        rp.fansi_print("DONE MAKING SAMPLE PREVIEW VIDEO!",'light blue green','underlined')

    #prompt=sample.instance_prompt
    downtemp_noise = ryan_dataset.downtemp_noise(
        sample_noise,
        noise_downtemp_interp=noise_downtemp_interp,
    )
    downtemp_noise = downtemp_noise[None]
    downtemp_noise = nw.mix_new_noise(downtemp_noise, degradation)

    assert downtemp_noise.shape == (B, F, C, H, W), (noise.shape,(B, F, C, H, W))

    if image is None            : sample_image = rp.as_pil_image(rp.as_numpy_image(sample_video[0].float()/2+.5))
    elif isinstance(image, str) : sample_image = rp.as_pil_image(rp.as_rgb_image(rp.load_image(image)))
    else                        : sample_image = rp.as_pil_image(rp.as_rgb_image(image))

    metadata = gather_vars('sample_path degradation downtemp_noise sample_gif_path sample_video sample_noise noise_downtemp_interp')
    settings = gather_vars('num_inference_steps guidance_scale v2v_strength')

    if noise  is None: noise  = downtemp_noise
    if video  is None: video  = sample_video
    if image  is None: image  = sample_image
    if prompt is None: prompt = sample_prompt

    assert noise.shape == (B, F, C, H, W), (noise.shape,(B, F, C, H, W))

    return gather_vars('prompt noise image video metadata settings')

def dict_to_name(d=None, **kwargs):
    """
    Used to generate MP4 file names
    
    EXAMPLE:
        >>> dict_to_name(dict(a=5,b='hello',c=None))
        ans = a=5,b=hello,c=None
        >>> name_to_dict(ans)
        ans = {'a': '5', 'b': 'hello', 'c': 'None'}
    """
    if d is None:
        d = {}
    d.update(kwargs)
    return ",".join("=".join(map(str, [key, value])) for key, value in d.items())

def name_to_dict(name):
    """
    Useful for analyzing output MP4 files

    EXAMPLE:
        >>> dict_to_name(dict(a=5,b='hello',c=None))
        ans = a=5,b=hello,c=None
        >>> name_to_dict(ans)
        ans = {'a': '5', 'b': 'hello', 'c': 'None'}
    """
    output=rp.as_easydict()
    for entry in name.split(','):
        key,value=entry.split('=',maxsplit=1)
        output[key]=value
    return output


def get_output_path(pipe, cartridge, subfolder:str, output_root:str):
    """
    Generates a unique output path for saving a generated video.

    Args:
        pipe: The video generation pipeline used.
        cartridge: Data used for generating the video.
        subfolder (str): Subfolder for saving the video.
        output_root (str): Root directory for output videos.

    Returns:
        String representing the unique path to save the video.
    """

    time = millis()

    output_name = (
        dict_to_name(
            t=time,
            pipe=pipe.pipe_name,
            lora=pipe.lora_name,
            steps    =               cartridge.settings.num_inference_steps,
            strength =               cartridge.settings.v2v_strength,
            degrad   =               cartridge.metadata.degradation,
            downtemp =               cartridge.metadata.noise_downtemp_interp,
            samp     = get_file_name(get_parent_folder(cartridge.metadata.sample_path), False),
        )
        + ".mp4"
    )

    output_path = get_unique_copy_path(
        path_join(
            make_directory(
                path_join(output_root, subfolder),
            ),
            output_name,
        ),
    )

    fansi_print(f"OUTPUT PATH: {rp.fansi_highlight_path(output_path)}", "blue", "bold")

    return output_path

def run_pipe(
    pipe,
    cartridge,
    subfolder="first_subfolder",
    output_root: str = "infer_outputs",
):
    output_mp4_path = get_output_path(pipe, cartridge, subfolder, output_root)
    
    if pipe.is_i2v:
        image = cartridge.image
        if isinstance(image, str):
            image = rp.as_pil_image(rp.load_image(image, use_cache=True))

    if pipe.is_v2v:
        print("Making v2v video...")
        v2v_video=cartridge.video
        v2v_video=rp.as_numpy_images(v2v_video) / 2 + .5
        v2v_video=rp.as_pil_images(v2v_video)

    video = pipe(
        prompt=cartridge.prompt,
        **(dict(image   =image                          ) if pipe.is_i2v else {}),
        **(dict(strength=cartridge.settings.v2v_strength) if pipe.is_v2v else {}),
        **(dict(video   =v2v_video                      ) if pipe.is_v2v else {}),
        num_inference_steps=cartridge.settings.num_inference_steps,
        latents=cartridge.noise.to(pipe.device),

        # FYI, SOME OTHER DEFAULT VALUES:
        # num_videos_per_prompt=1,
        # num_frames=num_frames,
        guidance_scale=cartridge.settings.guidance_scale,
        # generator=torch.Generator(device=device).manual_seed(42),
    ).frames[0]

    export_to_video(video, output_mp4_path, fps=8)

    sample_gif=load_video(cartridge.metadata.sample_gif_path)
    video=as_numpy_images(video)
    prevideo = horizontally_concatenated_videos(
        resize_list(sample_gif, len(video)),
        video,
        origin='bottom right',
    )
    import textwrap
    prevideo = rp.labeled_images(
        prevideo,
        position="top",
        labels=cartridge.metadata.sample_path +"\n"+output_mp4_path +"\n\n" + rp.wrap_string_to_width(cartridge.prompt, 250),
        size_by_lines=True,
        text_color='light light light blue',
        # font='G:Lexend'
    )

    preview_mp4_path = output_mp4_path + "_preview.mp4"
    preview_gif_path = preview_mp4_path + ".gif"
    print(end=f"Saving preview MP4 to preview_mp4_path = {preview_mp4_path}...")
    rp.save_video_mp4(prevideo, preview_mp4_path, framerate=16, video_bitrate="max", show_progress=False)
    compressed_preview_mp4_path = rp.save_video_mp4(prevideo, output_mp4_path + "_preview_compressed.mp4", framerate=16, show_progress=False)
    print("done!")
    print(end=f"Saving preview gif to preview_gif_path = {preview_gif_path}...")
    rp.convert_to_gif_via_ffmpeg(preview_mp4_path, preview_gif_path, framerate=12,show_progress=False)
    print("done!")

    return gather_vars('video output_mp4_path preview_mp4_path compressed_preview_mp4_path cartridge subfolder preview_mp4_path preview_gif_path')


# #prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
# prompt = "An old house by the lake with wooden plank siding and a thatched roof"
# prompt = "Soaring through deep space"
# prompt = "Swimming by the ruins of the titanic"
# prompt = "A camera flyby of a gigantic ice tower that a princess lives in, zooming in from far away from the castle into her dancing in the window"
# prompt = "A drone flyby of the grand canyon, aerial view"
# prompt = "A bunch of puppies running around a front lawn in a giant courtyard "
# #image = load_image(image=download_url_to_cache("https://media.sciencephoto.com/f0/22/69/89/f0226989-800px-wm.jpg"))

def main(
    lora_name='I2V5B_i2v_webvid_i3200',
    pipe_name=None,
    device=None,
    output_root='infer_outputs',
    subfolder='default_subfolder',

    #BROADCASTABLE:
    sample_path=None,
    degradation=0,
    noise_downtemp_interp='nearest',
    image=None,
    prompt=None,
    num_inference_steps=30,
    guidance_scale=6,
    v2v_strength=.5,#Timestep for when using Vid2Vid. Only set to not none when using a T2V model!
):
    """
    Main function to run the video generation pipeline with specified parameters.

    Args:
        pipe_name (str): Name of the pipeline to use ('T2V5B', 'T2V2B', 'I2V5B').
        lora_name (str): Name of the LoRA weights to load.
        device (str or int, optional): Device to run the model on (e.g., 'cuda:0' or 0).
        output_root (str): Root directory where output videos will be saved.
        subfolder (str): Subfolder within output_root to save outputs.
        sample_path (str or list, optional): Broadcastable. Path(s) to the sample `.pkl` file(s).
        degradation (float or list): Broadcastable. Degradation level(s) for the noise warp (float between 0 and 1).
        noise_downtemp_interp (str or list): Broadcastable. Interpolation method(s) for down-temporal noise. Options: 'nearest', 'blend', 'blend_norm'.
        image (str, PIL.Image, or list, optional): Broadcastable. Image(s) to use as the initial frame(s). Can be a URL or a path to an image.
        prompt (str or list, optional): Broadcastable. Text prompt(s) for video generation.
        num_inference_steps (int or list): Broadcastable. Number of inference steps for the pipeline.
    """

    if device is None:
        device = rp.select_torch_device(reserve=True, prefer_used=True)
        fansi_print(f"Selected torch device: {device}")


    cartridge_kwargs = rp.broadcast_kwargs(
        rp.gather_vars(
            "sample_path",
            "degradation",
            "noise_downtemp_interp",
            "image",
            "prompt",
            "num_inference_steps",
            "guidance_scale",
            "v2v_strength",
        )
    )
    rp.fansi_print("cartridge_kwargs:", "cyan", "bold")
    print(
        rp.indentify(
            rp.with_line_numbers(
                rp.fansi_pygments(
                    rp.autoformat_json(cartridge_kwargs),
                    "json",
                ),
                align=True,
            )
        ),
    )

    # cartridges = [load_sample_cartridge(**x) for x in cartridge_kwargs]
    cartridges = rp.load_files(lambda x:load_sample_cartridge(**x), cartridge_kwargs, show_progress='eta:Loading Cartridges')
    pipe = get_pipe(pipe_name, lora_name, device)

    output=[]
    for cartridge in cartridges:
        pipe_out = run_pipe(
            pipe=pipe,
            cartridge=cartridge,
            output_root=output_root,
            subfolder=subfolder,
        )

        output.append(
            rp.as_easydict(
                rp.gather(
                    pipe_out,
[
'output_mp4_path',
'preview_mp4_path',
'compressed_preview_mp4_path',
'preview_mp4_path',
'preview_gif_path',
],
                    as_dict=True,
                )
            )
        )

    return output

if __name__ == '__main__':
    import fire
    fire.Fire(main)



if False:
    #Some code I write for myself as a reference. Maybe run it in rp.

    import ryan_infer

    ryan_infer.main(
        sample_path=[
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ahead_job.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/agile_lent.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/agile_wing.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ajar_payer.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/alive_smog.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/amuse_chop.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/argue_life.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/bless_life.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/blog_voice.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/blunt_clay.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/blunt_swab.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/busy_proof.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/carve_stem.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/chomp_shop.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/clump_grub.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/agile_train.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ahead_shred.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/alien_wagon.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/arise_clear.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/balmy_fetch.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/balmy_rerun.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/balmy_smash.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/bleak_skier.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/bless_banjo.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/brisk_stump.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/ajar_doll.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/brisk_rug.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/ajar_clasp.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/bless_scan.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/both_cramp.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/clap_patch.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/amend_shred.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/clink_grief.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/droop_fever.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/elude_barge.pkl",
            
        ],
        # lora_name="I2V5B_i2v_webvid_i3200",
        lora_name="T2V5B_blendnorm_i16800_envato",
        # lora_name=None,
        degradation=1,
        noise_downtemp_interp="blend_norm",
        subfolder="abusing_nolora_blendnorm",
    )

if False:
    #Some code I made to turn the previous block into comparison videos...

    from rp import *


    def get_sample_name(path):
        # Return something like alive_smog from /root/CleanCode/Github/cogvideox-factory/infer_outputs/I2V5B_resum_blendnorm_i5400_webvid__infer=.5degrad/t=1730057038048,pipe=I2V5B,lora=I2V5B_resum_blendnorm_i5400_webvid,steps=30,degrad=0.5,downtemp=blend_norm,samp=alive_smog.mp4
        return path.split("=")[-1].split(".")[0]


    folder_paths, titles = list_transpose(
        [
            # ["/root/CleanCode/Github/cogvideox-factory/infer_outputs/T2V5B_RDeg_L2048_i4800__infer=.5degrad", ""],
            # ["/root/CleanCode/Github/cogvideox-factory/infer_outputs/T2V2B_RDeg_i30000__infer=.5degrad", ""],
            # ["/root/CleanCode/Github/cogvideox-factory/infer_outputs/t2v5b_tests", ""],
            ["/root/CleanCode/Github/cogvideox-factory/infer_outputs/i2v5b_nolora_nowarp", "Original CogVidX I2V Model (No LoRA or noisewarping) WITH Prompts"],
            ["/root/CleanCode/Github/cogvideox-factory/infer_outputs/I2V5B_nolora_webvid___infer___noprompt", "Original CogVidX I2V Model (No LoRA or noisewarping) WITH NO Prompts"],
            ["/root/CleanCode/Github/cogvideox-factory/infer_outputs/I2V5B_resum_blendnorm_i7600_webvid___infer___degrad=.5", "Normalized Blended Noise WITH Prompts"],
            ["/root/CleanCode/Github/cogvideox-factory/infer_outputs/I2V5B_resum_blendnorm_i7600_webvid___infer___degrad=.5,noprompt", "Normalized Blended Noise WITH NO Prompts"],
        ]
    )

    folder_names = get_folder_names(folder_paths)
    titles = [t + ": " + fn for fn, t in zip(folder_names, titles)]

    mp4s = rp_glob(x + "/*.mp4" for x in folder_paths)
    mp4s = [x for x in mp4s if not x.endswith(".mp4_preview.mp4")]
    mp4_bundles = cluster_by_key(mp4s, get_sample_name)
    mp4_bundles = [x for x in mp4_bundles if len(x) == len(folder_paths)]

    def process_bundle(mp4_bundle, show_progress=False):
        for i in range(len(mp4_bundle)):
            mp4_bundle[i] += "_preview.mp4"
        sample_name = get_sample_name(mp4_bundle[0])  # Index doesnt matter
        videos = load_videos(mp4_bundle, show_progress=show_progress, use_cache=True)
        videos = labeled_videos(videos, get_file_names(mp4_bundle), background_color="dark green")
        videos = labeled_videos(videos, titles, size=25, background_color="dark green")
        preview_video = vertically_concatenated_videos(videos)

        preview_file = sample_name + ".mp4"
        preview_file = get_unique_copy_path(preview_file)
        output_mp4 = save_video_mp4(preview_video, preview_file, show_progress=show_progress, framerate=12)
        fansi_print("SAVED " + fansi_highlight_path(output_mp4), "green")
        return output_mp4

    # Output Directory
    output_dir = "/root/CleanCode/Github/cogvideox-factory/untracked/comparison_outputs"
    if not folder_is_empty(output_dir):
        output_dir = get_unique_copy_path(output_dir)
    take_directory(output_dir)
    fansi_print(f"output_dir = {fansi_highlight_path(output_dir)}", "cyan", "bold")

    load_files(process_bundle, mp4_bundles, show_progress=True, num_threads=None)

if False:
    #Some code to sequentially go through a long video with I2V. Remember to first get all the cartridges!

    import ryan_infer
    import rp


    sample_paths = [
        "/root/CleanCode/Github/cogvideox-factory/datasets/factory_scene_snippets/speed=1__reversed=True/00.mp4.pkl",
        "/root/CleanCode/Github/cogvideox-factory/datasets/factory_scene_snippets/speed=1__reversed=True/01.mp4.pkl",
        "/root/CleanCode/Github/cogvideox-factory/datasets/factory_scene_snippets/speed=1__reversed=True/02.mp4.pkl",
        "/root/CleanCode/Github/cogvideox-factory/datasets/factory_scene_snippets/speed=1__reversed=True/03.mp4.pkl",
        "/root/CleanCode/Github/cogvideox-factory/datasets/factory_scene_snippets/speed=1__reversed=True/04.mp4.pkl",
        "/root/CleanCode/Github/cogvideox-factory/datasets/factory_scene_snippets/speed=1__reversed=True/05.mp4.pkl",
        "/root/CleanCode/Github/cogvideox-factory/datasets/factory_scene_snippets/speed=1__reversed=True/06.mp4.pkl",
        "/root/CleanCode/Github/cogvideox-factory/datasets/factory_scene_snippets/speed=1__reversed=True/07.mp4.pkl",
    ]


    title = "factory"
    lora_name = "I2V5B_resum_blendnorm_i13400_webvid"
    subfolder = f'ZSEQUENTIAL__{lora_name}__{title}__{format_current_date("EST")}'
    initial_image = "/root/CleanCode/Github/cogvideox-factory/datasets/EVPG_Camera_Control_Style_001.jpg"

    image = initial_image


    for sample_path in sample_paths:
        result = ryan_infer.main(
            sample_path=sample_path,
            image=image,
            lora_name=lora_name,
            subfolder=subfolder,
            pipe_name="I2V5B",
            noise_downtemp_interp="blend_norm",
            degradation=0.5,
            num_inference_steps=50,
            # prompt='',
        )

        video_path = result[0].output_mp4_path

        last_frame = rp.load_video(video_path)[-1]

        image = save_image_jpg(last_frame, video_path + ".png")

        fansi_print("SAVED IMAGE TO " + image, "yellow green", "bold italic", "dark red")


#FOR VPS
# Some code I write for myself as a reference. Maybe run it in rp.

    import ryan_infer

    ryan_infer.main(
        sample_path=[
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___wiry_royal.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___broke_chill.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___icy_robe.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___humid_squat.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___bony_tug.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___repel_ozone.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___hurt_shore.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___camera_tt_180__dynamic_XXXX__light_tt_180__first_49_frames___poach_flock.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___swear_video.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___amuse_ion.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___ahead_set.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___undo_slick.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___avoid_thumb.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___cure_power.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___icy_maker.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___camera_tt_180__dynamic_XXXX__light_tt_180__first_49_frames___slurp_image.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___amend_year.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___clap_cadet.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___showy_crowd.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___tweak_ivy.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___brisk_rush.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___wipe_femur.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___ashen_theme.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___camera_tt_180__dynamic_XXXX__light_tt_180__first_49_frames___carve_park.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___late_wick.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___humid_lurch.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___wiry_broom.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___happy_grit.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___swear_view.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___agile_heat.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___gt_compare_seq__pose_XXXX__fixed_light_74__middle_49_frames___crave_kilt.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___utter_stole.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___wiry_royal.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__KW___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___task_spoon.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___broke_chill.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__KW___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___glad_front.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___icy_robe.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__KW___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___thank_blank.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___humid_squat.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__KW___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___bask_tilt.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___bony_tug.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__KW___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___amuse_net.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___repel_ozone.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__KW___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___enter_sugar.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/KW___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___hurt_shore.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__KW___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___yelp_basil.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___camera_tt_180__dynamic_XXXX__light_tt_180__first_49_frames___poach_flock.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__MC___camera_tt_180__dynamic_XXXX__light_tt_180__first_49_frames___hasty_genre.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___swear_video.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__MC___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___mousy_frost.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___amuse_ion.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__MC___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___nutty_push.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___ahead_set.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__MC___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___mousy_sniff.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___undo_slick.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__MC___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___utter_grunt.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___avoid_thumb.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__MC___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___elude_mural.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___cure_power.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__MC___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___dried_buggy.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/MC___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___icy_maker.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__MC___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___relax_acorn.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___camera_tt_180__dynamic_XXXX__light_tt_180__first_49_frames___slurp_image.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__NL___camera_tt_180__dynamic_XXXX__light_tt_180__first_49_frames___fax_lever.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___amend_year.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__NL___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___grasp_hash.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___clap_cadet.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__NL___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___enter_wrist.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___showy_crowd.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__NL___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___ritzy_ice.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___tweak_ivy.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__NL___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___thud_scan.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___brisk_rush.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__NL___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___elope_deed.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___wipe_femur.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__NL___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___poach_latch.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/NL___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___ashen_theme.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__NL___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___spew_canon.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___camera_tt_180__dynamic_XXXX__light_tt_180__first_49_frames___carve_park.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__ZK___camera_tt_180__dynamic_XXXX__light_tt_180__first_49_frames___neat_slab.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___late_wick.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__ZK___camera_tt_180__dynamic_XXXX__light_tt_180__squashed_to_49_frames___erupt_scoop.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___humid_lurch.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__ZK___camera_tt_180__pose_XXXX__camera_space_right__first_49_frames___quiet_goofy.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___wiry_broom.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__ZK___camera_tt_180__pose_XXXX__camera_space_right__squashed_to_49_frames___dizzy_drove.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___happy_grit.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__ZK___circle_camera__dynamic_XXXX__light_tt_360__first_49_frames___enter_dock.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___swear_view.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__ZK___circle_camera__dynamic_XXXX__light_tt_360__squashed_to_49_frames___elope_stick.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___agile_heat.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__ZK___gt_compare_seq__pose_XXXX__fixed_light_74__first_49_frames___etch_agent.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___gt_compare_seq__pose_XXXX__fixed_light_74__middle_49_frames___crave_kilt.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__ZK___gt_compare_seq__pose_XXXX__fixed_light_74__middle_49_frames___clap_crook.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/ZK___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___utter_stole.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Testing_Samples/rot90__ZK___gt_compare_seq__pose_XXXX__fixed_light_74__squashed_to_49_frames___clump_buzz.pkl",
        ],
        prompt=[
            "The video features a man with a beard and long hair tied back, wearing a dark shirt, and he appears in a neutral setting with plain, soft lighting. His expressions progress from a neutral side profile, gently transitioning to facing forward with a direct gaze and slightly parted lips, suggesting that he might be speaking or preparing to speak. The camera smoothly pans from a profile view to a frontal view and back to an opposite profile, capturing subtle changes in his facial expressions and maintaining a consistent shallow depth of field that emphasizes his features against a simple background.",
            "The video features a serene portrait of a bearded man with long, tied-back hair, wearing a black shirt. His gaze shifts subtly from the side to directly forward, maintaining a calm and neutral expression throughout. The lighting is soft and even, highlighting the texture of his beard and slight skin markings. The background is a clean white, ensuring focus remains on his facial features. The camera remains steady, capturing a gradual pan from a side profile to a frontal view, emphasizing his composed demeanor and thoughtful presence.",
            "The video features a man with a full beard and long hair tied back, wearing a black t-shirt. His expression is calm and neutral, eyes gazing forward while occasionally shifting slightly. The lighting is soft and even, creating a clear and natural look. The camera smoothly captures his face from various angles, including side profiles and a frontal view, showcasing his contemplative demeanor without any speech or dramatic change in expression. The focus remains on his facial features and expression throughout the sequence.",
            "The video depicts a man with a beard and mustache wearing a black shirt against a white background. The lighting is bright and even, casting no shadows. His expression transitions from neutral to contemplative as he looks upwards. Throughout the frames, his lips slightly part as if beginning to speak, but he shows minimal emotional change. The camera smoothly tilts upwards, following his gaze, focusing closely on his face and capturing subtle facial details and expressions.",
            "The video features a bearded man with tied-back hair, wearing a black shirt, who is initially expressionless. As the scene progresses, he looks upwards and slightly to the side, with his lips parting as if he's speaking or about to speak. His expressions shift subtly, suggesting contemplation or engagement in thought. The lighting is bright and even, casting no shadows, and the background is a plain white, emphasizing his features. The camera maintains a steady close-up on his face, but slightly shifts angles, focusing on different facial features, adding dynamic movement to the otherwise static scene.",
            "The video features a man with long dark hair tied back and a full beard, wearing a black t-shirt, displaying a joyful expression throughout. He smiles broadly, exuding warmth and contentment. His eyes show happiness as he looks slightly upwards and to the side, capturing different angles of his face. The scene is shot in bright, even lighting, emphasizing the subject's facial features against a plain white background. The camera smoothly transitions between close-up and side profile views, highlighting his genuine and relaxed demeanor.",
            "The video features a bearded man with shoulder-length hair tied back, wearing a black shirt. His expressions convey a sense of happiness and contentment, with a broad smile throughout. He occasionally shifts his gaze, looking forward and slightly upward, with a relaxed demeanor. The lighting is bright, creating a high-key effect with minimal shadows, suggesting a clean, studio-like environment. The camera maintains a steady focus with close-up and medium shots, capturing multiple angles of the man's smiling face and subtle facial movements.",
            "The video features a woman with her hair in a ponytail, wearing a sleeveless maroon top and earrings. She is expressive, showcasing a range of emotions from contemplative to slightly frustrated, as seen in her closed eyes and furrowed brow. Her mouth movements suggest she is speaking, and her body and head subtly shift angles. The lighting is soft and even, highlighting her expressions and details of her face, while the background is bright and unobtrusive. The camera remains steady, focusing mainly on her profile and slightly turning to capture different angles of her face.",
            "The woman in the video, with her hair tied back and wearing a sleeveless maroon top, appears engaged and expressive as she talks. Her facial expressions transition from serious to slightly smiling, exhibiting confidence and intensity. She wears subtle makeup and jewelry that include stud earrings and a delicate necklace. The close-up shots emphasize her face against a bright, uniform background, with soft, diffuse lighting highlighting her facial features. The camera angle shifts between side and frontal views, capturing her articulate movements and steady gaze, suggesting a conversation or speech delivery.",
            "The video depicts a woman with a ponytail wearing a mauve sleeveless top and small earrings. Her expression shifts from neutral to slightly contemplative as the camera transitions smoothly from side profile to front view. The lighting is bright and even, highlighting her facial features. Her eyes are subtly lined, enhancing her focused gaze, and her demeanor remains calm and composed throughout the video.",
            "The video showcases a woman with long brown hair tied in a ponytail and wearing a sleeveless burgundy top and small earrings. Her expressions are calm and composed as she moves from a profile view to a slight rotation, allowing a frontal perspective. The lighting is bright and even, suggesting a studio setting with a plain white background. Her makeup is subtle, highlighting her eyes, and she maintains a neutral expression throughout, without speaking. The camera remains steady, focusing on her face and capturing her transitions between different angles smoothly.",
            "The video features a woman wearing a maroon sleeveless top and heart-shaped necklace, with her hair pulled back. Her expressions progress from slight concern to a more animated, emotive state. She seems engaged in conversation, her mouth opening and closing as if speaking with varying intensity. The lighting is bright and even, creating a soft, clear background. Her eyebrows occasionally furrow, and her eyes widen, adding emphasis to her expressions. The camera remains steady, maintaining a close-up focus on her face, capturing subtle changes in her emotion and expression throughout the sequence.",
            "The video features a woman with her hair tied back, wearing a sleeveless maroon top and a heart necklace, engaged in a conversation. Her expressions are a mix of concern and determination, accompanied by subtle movements and articulate gestures, indicating she is thoughtfully explaining something. Her eyes are expressive, and she appears to be in a calm and controlled mood, as she looks up and to the side while speaking. The lighting is bright and even, emphasizing her facial features. The camera remains steady with mild close-up shots, focusing on capturing her emotions and expressions clearly against a plain background, giving the scene an intimate and direct feel, as if she is addressing the viewer or someone off-camera.",
            "The video showcases a woman with long dark hair tied back, wearing a sleeveless maroon top and small earrings, exuding a joyful and content demeanor. Her expressions transition from a broad smile to a subtle, contemplative gaze, maintaining a warm and pleasant vibe throughout. The lighting is bright and even, emphasizing her features and the white background creates a clean, neutral setting. The camera provides close-up and slightly angled shots, capturing her face from different perspectives without significant movement. Her eyes occasionally glance around softly, reflecting a calm and relaxed composure.",
            "The video features a woman with long dark hair tied back, wearing a sleeveless maroon top and small diamond earrings. She appears in a series of close-up shots with a neutral to cheerful demeanor, often smiling gently and looking around. Subtle expressions, such as slight eyebrow raises, add a touch of curiosity or interest. There is a soft, even lighting that highlights her features and creates a clean, bright background. The camera angles shift slightly, including frontal and side profiles, to capture different aspects of her expressions and presence, maintaining a steady focus on her face throughout.",
            "The video features a young man with curly hair, wearing a casual gray shirt, set against a bright white background. He starts with a closed-mouth smile, then closes his eyes and appears to be softly smiling or chuckling, indicating a sense of ease or amusement. His expressions transition from contemplative to more relaxed and joyful as his eyes reopen. The lighting remains consistent and soft, highlighting his facial features, and the camera maintains a static, close-up, side-angle shot, allowing viewers to focus on the subtle changes in his emotional expressions.",
            "The video features a man with curly hair and a short beard, wearing a gray shirt, in a brightly lit environment with a plain white background. Initially seen in profile, he gradually turns towards the camera, exhibiting a warm, friendly smile and appearing engaged and content. His facial expressions suggest he is speaking, with natural and relaxed movements. The lighting is soft and consistent, highlighting his features. The camera remains steady, capturing him from different angles as he turns, effectively focusing on his expressions and demeanor.",
            "The video features a man with curly hair and a goatee, wearing a gray t-shirt. He maintains a neutral expression throughout, with a slight frown and focused gaze that suggests concentration or contemplation. The lighting is bright and even, highlighting his features against a plain white background. There is a stationary camera setup that captures a profile view, slowly transitioning to a front-facing view, emphasizing his facial expression without any visible movement or speaking from the subject.",
            "The video features a young man with curly hair and a short beard, wearing a gray shirt, in a minimalistic setting with soft, even lighting. Initially, the man is seen in profile with a neutral expression, gazing straight ahead. As the camera angle shifts slightly, his face becomes more visible, showing a subtle furrow in his brow that suggests a contemplative or focused mood. The man remains composed and introspective, with no visible change in facial expression. The camera continues to capture various angles of his face, maintaining a smooth and steady focus on his features. Throughout the video, the lighting remains consistent, highlighting his facial expressions without casting harsh shadows.",
            "The video portrays a young man with curly hair and a short beard, wearing a gray t-shirt against a plain white background. His facial expressions transition from a serious, contemplative look to speaking with emphasis, showcasing slight frustration or determination in his eyes and slightly furrowed brows. His mouth opens wider as he speaks more intently, suggesting he is making a point or explaining something with conviction. The lighting is bright and even, highlighting his facial features with no shadows, indicating a studio setting. The camera remains steady, focusing on his expressions and speech, capturing a range of emotions through close-up shots.",
            "The video features a young man with curly hair, wearing a gray shirt, in an isolated, well-lit setting. He appears thoughtful and introspective, his expressions shifting from focused concentration to a more contemplative gaze. Initially, his brows are furrowed with slight tension, suggesting he's pondering a complex issue. As the frames progress, his gaze shifts upwards as if seeking insight or inspiration from above. His facial expressions convey a narrative of contemplation and resolution, moving from initial concern to a calm acceptance. The camera remains steady, focusing tightly on his face, emphasizing his emotions and expressions. The bright, even lighting highlights his features against a neutral background, drawing attention to subtle changes in his expression.",
            "The video showcases a man with curly hair wearing a gray shirt, who is experiencing a range of subtle, humorous expressions. His eyes are tightly shut, and his face is scrunched up in a grimace, suggesting discomfort or anticipation. The lighting is bright, creating an evenly lit, clean white background that highlights his facial features. The sequence captures his head moving slightly, possibly indicating a reaction to an unseen stimulus. His expressions remain tense throughout, with no dialogue, as the camera maintains a close-up shot focusing on his expressions.",
            "The video portrays a young man with curly hair and a goatee, wearing a gray shirt in a well-lit, neutral background. His expressions shift from neutral to an intense grimace, with his eyes tightly closed and face scrunched up as if reacting to a strong sensation or emotion. The camera captures him from various angles, including front and profile views, emphasizing his deep concentration and possibly discomfort. The lighting remains consistent and bright, focusing on his facial features as he undergoes this experience.",
            "The video features a woman with blonde hair tied back, wearing a brown shirt. Her expressions transition smoothly from attentive listening to expressive speaking, indicating a dynamic conversation. Initially, her mouth is slightly pursed, then she begins articulating, her eyes conveying engagement and intensity. The lighting is bright and even, highlighting her features against a plain background, giving a clean, focused aesthetic. The camera remains steady, capturing a profile and then a full face view, emphasizing her verbal expressions and subtle changes in mood during the dialogue.",
            "The video features a woman with light brown hair tied back, wearing a brown shirt. She appears engaged in a thoughtful conversation, her facial expressions shifting from attentiveness to subtle surprise and mild concern. Her eyes are expressive, suggesting deep focus as she speaks, with her lips moving gently, indicating dialogue. The camera angle captures her from the side and slightly turns to show more of her face, maintaining a crisp, well-lit setting with a white background that emphasizes her features. The lighting is soft and even, highlighting her skin tone and expressions, adding a calm and professional atmosphere to the dialogue.",
            "The video features a woman with light hair tied back, wearing a brown top, and displaying a calm, focused expression. Her face is seen in profile and gradually turns towards the camera, capturing her neutral emotions throughout. The lighting is soft and even, creating a bright and clear atmosphere. The camera remains static, emphasizing her facial features and subtle expressions without any movement. Her demeanor is composed and serene, adding an understated elegance to the visuals.",
            "The video depicts a woman with light hair pulled back, wearing a brown shirt. Her expressions remain calm and neutral as she slightly turns her head from a profile view to a more central position. The lighting is bright and evenly distributed, providing a soft focus on her face, and the background is white, emphasizing her features. Her eyes appear contemplative and her lips are slightly pursed, suggesting a serene or introspective mood. There is no significant camera movement, maintaining a steady, close-frame view throughout the series of shots.",
            "The video showcases a close-up of a woman with light hair tied back, wearing a brown shirt, against a plain white background. Her facial expressions evolve from a serious, contemplative look to one of concern and emotion, with her gaze directed upward at various points. She appears to be speaking, as her mouth moves through open and closed positions. The lighting is soft and even, highlighting the emotional nuances of her face. The camera remains steady, focusing intently on her expressions, creating an intimate and personal atmosphere throughout the sequence.",
            "The video features a woman with light hair pulled back, wearing a brown top, in an emotional monologue. Her expressions transition from serious and thoughtful to slightly upset, with furrowed brows and pursed lips, suggesting contemplation or distress. She occasionally looks upwards, possibly reflecting or recalling something significant. Her eyes close briefly, enhancing the feeling of sincerity or deep emotion. The setting is brightly lit with a soft, even light, creating a sense of clarity and focus on her facial expressions. The camera remains steady, capturing her head and shoulders, emphasizing the intensity of her emotions and the rawness of the moment.",
            "The video captures a woman with light blonde hair pulled back into a ponytail, wearing a brown shirt, in a brightly lit setting with a seamless white background. Her expression is sad, characterized by a deeply furrowed brow and downward-cast lips, as she looks forward and slightly to the side. The camera remains steady and primarily focuses on her face, highlighting her pensive demeanor through close-up shots from various angles. The lighting is even and soft, eliminating any shadows, and emphasizing the subtle nuances of her facial features and emotional expression. Her eyes occasionally shift, reflecting introspection, while her overall posture remains still and slightly slouched, conveying a sense of melancholy throughout the sequence.",
            "The video features a woman with light hair pulled back, wearing a brown shirt. Her expression is consistently one of concern or worry, with furrowed brows and downturned lips. The lighting is bright and even, creating a clear view of her emotional state. Throughout the video, the camera remains steady, capturing her gradually turning her head to the side. She remains silent, and her gaze is directed downward, adding to the somber mood. The white background and her neutral attire emphasize her expressions as the focal point.",
            "The video features a woman with light hair pulled back, wearing a brown shirt. Her expression transitions into a pout with a furrowed brow, indicating sadness or disappointment. The lighting is bright and even, highlighting her facial features with no shadows, suggesting a studio setting. She looks straight ahead, then to her left, and slightly downwards, maintaining the pout throughout, which further accentuates her emotional state. The camera remains steady, capturing her expressions closely and consistently, emphasizing her emotional demeanor without any distractions.",
            "The video features a bearded man with long hair tied back, wearing a dark shirt. His expressions gradually shift from a neutral gaze upward to a contemplative look as he moves his head slightly up and down. The lighting is soft, creating a gentle contrast on his face. The camera remains steady, capturing close-up shots of his face and subtle movements. His demeanor shifts from introspective to slightly engaged, possibly indicating he is speaking or reacting thoughtfully. The background is a plain, well-lit white, emphasizing his facial features and expressions.",
            "The video features a bearded man with long hair tied back, wearing a dark shirt. His expressions gradually shift from a neutral gaze upward to a contemplative look as he moves his head slightly up and down. The lighting is soft, creating a gentle contrast on his face. The camera remains steady, capturing close-up shots of his face and subtle movements. His demeanor shifts from introspective to slightly engaged, possibly indicating he is speaking or reacting thoughtfully. The background is a plain, well-lit white, emphasizing his facial features and expressions.",
            "The video depicts a man with a graying beard and long, slicked-back hair, lying on his back, wearing a black shirt. He appears calm and contemplative, as the camera slowly moves from a profile view to a more frontal angle, capturing his relaxed expression and closed-lipped demeanor. The lighting is bright and even, suggesting a clinical or sterile environment. The man remains still throughout the frames, with a subtle focus on his facial features. The overall mood is serene and introspective, with no evident dialogue or dramatic movements.",
            "The video depicts a man with a graying beard and long, slicked-back hair, lying on his back, wearing a black shirt. He appears calm and contemplative, as the camera slowly moves from a profile view to a more frontal angle, capturing his relaxed expression and closed-lipped demeanor. The lighting is bright and even, suggesting a clinical or sterile environment. The man remains still throughout the frames, with a subtle focus on his facial features. The overall mood is serene and introspective, with no evident dialogue or dramatic movements.",
            "The video features a bearded man with long hair tied back, wearing a black shirt, lying down as the camera gradually rotates to show different angles of his face. His expression is calm and relaxed, with minimal movement observed throughout. The lighting is neutral and soft, providing even illumination across his face, enhancing the details in his hair and beard. The video style is simple and focused, with smooth camera transitions following the natural contours of his face and maintaining clarity and focus.",
            "The video features a bearded man with long hair tied back, wearing a black shirt, lying down as the camera gradually rotates to show different angles of his face. His expression is calm and relaxed, with minimal movement observed throughout. The lighting is neutral and soft, providing even illumination across his face, enhancing the details in his hair and beard. The video style is simple and focused, with smooth camera transitions following the natural contours of his face and maintaining clarity and focus.",
            "The video features a man with a greying beard and slicked-back hair, wearing a black shirt against a bright, well-lit background. His expression transitions from neutral to slightly animated as he talks, suggesting conversation or explanation. The soft, even lighting highlights his facial features and the texture of his beard. The camera remains static, focusing closely on his upper body and face, capturing subtle changes in his expression and mouth movements as he speaks. The mood is calm and focused.",
            "The video features a man with a greying beard and slicked-back hair, wearing a black shirt against a bright, well-lit background. His expression transitions from neutral to slightly animated as he talks, suggesting conversation or explanation. The soft, even lighting highlights his facial features and the texture of his beard. The camera remains static, focusing closely on his upper body and face, capturing subtle changes in his expression and mouth movements as he speaks. The mood is calm and focused.",
            "The video features a man with a neatly groomed beard and tied back hair, wearing a black shirt. He appears to be in a bright, softly lit environment, which highlights his facial features. Throughout the video, he looks contemplative and engages in speaking some words, as suggested by his slightly moving lips. His facial expressions shift subtly, indicating thoughtfulness or explaining something. The camera remains focused on him with minimal movement, maintaining a consistent close-up shot, which enhances the personal and intimate feel of the recording.",
            "The video features a man with a neatly groomed beard and tied back hair, wearing a black shirt. He appears to be in a bright, softly lit environment, which highlights his facial features. Throughout the video, he looks contemplative and engages in speaking some words, as suggested by his slightly moving lips. His facial expressions shift subtly, indicating thoughtfulness or explaining something. The camera remains focused on him with minimal movement, maintaining a consistent close-up shot, which enhances the personal and intimate feel of the recording.",
            "The video showcases a person with a full beard and dark hair, wearing a black shirt, lying on their side against a bright white background. The person is smiling broadly throughout, conveying warmth and happiness. Their eyes slightly squint with the smile, adding to the authentic expression of joy. The lighting is even and bright, highlighting the facial features clearly. The individual appears relaxed and content. During the video, the camera remains stationary, focusing closely on the individual’s face, capturing the genuine and relaxed emotion without any noticeable movement or change in perspective.",
            "The video showcases a person with a full beard and dark hair, wearing a black shirt, lying on their side against a bright white background. The person is smiling broadly throughout, conveying warmth and happiness. Their eyes slightly squint with the smile, adding to the authentic expression of joy. The lighting is even and bright, highlighting the facial features clearly. The individual appears relaxed and content. During the video, the camera remains stationary, focusing closely on the individual’s face, capturing the genuine and relaxed emotion without any noticeable movement or change in perspective.",
            "The video features a bearded man wearing a black shirt, lying down with a cheerful expression. His face is relaxed and serene, as he continuously smiles and occasionally laughs with his eyes partly squinting in joy. Across the frames, the camera captures him from different angles, focusing on his face in close-up shots, which remain well-lit against a plain white background. The lighting is even, highlighting his features and beard subtly with no shadows, while the camera movements are gentle, smoothly transitioning between different perspectives of his smiling face.",
            "The video features a bearded man wearing a black shirt, lying down with a cheerful expression. His face is relaxed and serene, as he continuously smiles and occasionally laughs with his eyes partly squinting in joy. Across the frames, the camera captures him from different angles, focusing on his face in close-up shots, which remain well-lit against a plain white background. The lighting is even, highlighting his features and beard subtly with no shadows, while the camera movements are gentle, smoothly transitioning between different perspectives of his smiling face.",
            "The video shows a woman lying on her back in a bright, evenly lit setting. She is wearing a maroon sleeveless top and a delicate necklace. Her hair is tied back, and she wears stud earrings. Initially, her expression is calm and relaxed with eyes mostly closed, but she gradually appears to become more alert and engages in speaking, as her mouth opens and her eyes focus upward. Her facial expressions shift from serene to slightly intense, suggesting thought or concentration. The camera remains stationary, capturing her profile and gradually shifting to a more frontal view as she talks, creating a dynamic progression in her demeanor.",
            "The video shows a woman lying on her back in a bright, evenly lit setting. She is wearing a maroon sleeveless top and a delicate necklace. Her hair is tied back, and she wears stud earrings. Initially, her expression is calm and relaxed with eyes mostly closed, but she gradually appears to become more alert and engages in speaking, as her mouth opens and her eyes focus upward. Her facial expressions shift from serene to slightly intense, suggesting thought or concentration. The camera remains stationary, capturing her profile and gradually shifting to a more frontal view as she talks, creating a dynamic progression in her demeanor.",
            "The video showcases a woman in a maroon tank top lying on her back, transitioning to sitting up; her expressions evolve from calm to focused as she speaks, highlighting her engagement. Her hair is tied back in a ponytail, with a simple necklace and earrings adding to her appearance. She appears to articulate her thoughts with increasing intensity, marked by her changing facial expressions and lip movements. The lighting is bright and neutral, emphasizing her features against a plain background. The camera gradually shifts from a side profile to a frontal view, capturing her dynamic expressions and actions as she moves upright, further emphasizing her clarity and determination in the scene.",
            "The video showcases a woman in a maroon tank top lying on her back, transitioning to sitting up; her expressions evolve from calm to focused as she speaks, highlighting her engagement. Her hair is tied back in a ponytail, with a simple necklace and earrings adding to her appearance. She appears to articulate her thoughts with increasing intensity, marked by her changing facial expressions and lip movements. The lighting is bright and neutral, emphasizing her features against a plain background. The camera gradually shifts from a side profile to a frontal view, capturing her dynamic expressions and actions as she moves upright, further emphasizing her clarity and determination in the scene.",
            "The video shows a woman with long brown hair, wearing a maroon tank top and a thin necklace, lying on her back in a bright, evenly-lit setting with a white background. Her head turns gradually from a side profile to a more frontal view across the frames, with subtle facial expressions that remain calm and neutral. She wears small round earrings, and there are no drastic changes in her emotions. Her eyes are open, suggesting a state of relaxation or contemplation. The lighting remains soft and consistent, highlighting her facial features delicately. The camera smoothly transitions, likely with a gentle pan or tilt movement, maintaining a close-up shot that focuses on the subject's face and head.",
            "The video shows a woman with long brown hair, wearing a maroon tank top and a thin necklace, lying on her back in a bright, evenly-lit setting with a white background. Her head turns gradually from a side profile to a more frontal view across the frames, with subtle facial expressions that remain calm and neutral. She wears small round earrings, and there are no drastic changes in her emotions. Her eyes are open, suggesting a state of relaxation or contemplation. The lighting remains soft and consistent, highlighting her facial features delicately. The camera smoothly transitions, likely with a gentle pan or tilt movement, maintaining a close-up shot that focuses on the subject's face and head.",
            "The video captures a woman in a maroon sleeveless top, lying down and then gradually sitting up. She appears calm and relaxed with a neutral expression as she transitions. The lighting is bright and even, creating a soft appearance against a white background. The camera smoothly moves from a profile view to a more frontal perspective, highlighting her facial features and the slight movements in her expression. The absence of dramatic changes in lighting or emotion emphasizes a serene and focused atmosphere.",
            "The video captures a woman in a maroon sleeveless top, lying down and then gradually sitting up. She appears calm and relaxed with a neutral expression as she transitions. The lighting is bright and even, creating a soft appearance against a white background. The camera smoothly moves from a profile view to a more frontal perspective, highlighting her facial features and the slight movements in her expression. The absence of dramatic changes in lighting or emotion emphasizes a serene and focused atmosphere.",
            "The video features a woman with her hair tied back, wearing a maroon top and a delicate necklace, speaking while showing varied emotions. Her expressions range from pensive to slightly intense, with her mouth frequently moving, indicating she is talking. The lighting is bright and even, emphasizing her features against a white backdrop. The camera remains stable, focusing closely on her face, capturing her subtle changes in expression and speech. Her earrings add a touch of elegance to her appearance.",
            "The video features a woman with her hair tied back, wearing a maroon top and a delicate necklace, speaking while showing varied emotions. Her expressions range from pensive to slightly intense, with her mouth frequently moving, indicating she is talking. The lighting is bright and even, emphasizing her features against a white backdrop. The camera remains stable, focusing closely on her face, capturing her subtle changes in expression and speech. Her earrings add a touch of elegance to her appearance.",
            "In this video, a woman wearing a maroon top and a necklace with a heart-shaped pendant is speaking with a serious expression, her emotions conveyed through subtle changes in her facial expressions and head movements. She appears to be engaged in a thoughtful conversation, as indicated by her shifting gaze and slightly furrowed brows. Her dark hair is tied back, and she wears small stud earrings, adding to her poised appearance. The lighting is bright and even, highlighting her features against a neutral backdrop. The camera maintains a steady, close-up angle, focusing on her face to emphasize her verbal communication, suggesting an intimate and engaging dialogue.",
            "In this video, a woman wearing a maroon top and a necklace with a heart-shaped pendant is speaking with a serious expression, her emotions conveyed through subtle changes in her facial expressions and head movements. She appears to be engaged in a thoughtful conversation, as indicated by her shifting gaze and slightly furrowed brows. Her dark hair is tied back, and she wears small stud earrings, adding to her poised appearance. The lighting is bright and even, highlighting her features against a neutral backdrop. The camera maintains a steady, close-up angle, focusing on her face to emphasize her verbal communication, suggesting an intimate and engaging dialogue.",
            "The video features a woman with brown hair tied back, wearing a sleeveless, maroon top and a delicate necklace, smiling warmly with relaxed expressions. Her demeanor is joyful as she appears to be in a bright setting with soft, even lighting highlighting her friendly and engaging expression. The camera captures close-up angles of her face, emphasizing her smiling eyes, as it shifts gently to offer slightly varied perspectives. There is no indication of speech, as the focus remains on the warmth and genuineness of her smile throughout the sequence.",
            "The video features a woman with brown hair tied back, wearing a sleeveless, maroon top and a delicate necklace, smiling warmly with relaxed expressions. Her demeanor is joyful as she appears to be in a bright setting with soft, even lighting highlighting her friendly and engaging expression. The camera captures close-up angles of her face, emphasizing her smiling eyes, as it shifts gently to offer slightly varied perspectives. There is no indication of speech, as the focus remains on the warmth and genuineness of her smile throughout the sequence.",
            "The video features a woman with dark hair tied back, wearing a sleeveless burgundy top and a delicate necklace. She appears content and joyful, consistently smiling and exhibiting signs of amusement. Her facial expressions convey warmth and happiness as she slightly shifts her gaze. The background is a bright, even white, suggesting the presence of soft lighting that highlights her features. The camera maintains a steady close-up focus on her face, capturing her smiling transitions and subtle movements without any sudden shifts or dramatic angles.",
            "The video features a woman with dark hair tied back, wearing a sleeveless burgundy top and a delicate necklace. She appears content and joyful, consistently smiling and exhibiting signs of amusement. Her facial expressions convey warmth and happiness as she slightly shifts her gaze. The background is a bright, even white, suggesting the presence of soft lighting that highlights her features. The camera maintains a steady close-up focus on her face, capturing her smiling transitions and subtle movements without any sudden shifts or dramatic angles.",
            "The video features a man with curly hair and a goatee lying down, wearing a gray shirt. His facial expressions transition from calm and relaxed to a gentle smile and laughter, suggesting a pleasant or humorous thought. His eyes are mostly closed, then open slightly, conveying a sense of contentment and ease. The lighting is bright and even, creating a soft focus on his face against a white background. The camera remains steady, focusing on his profile, capturing his subtle emotional shifts with clarity.",
            "The video features a man with curly hair and a goatee lying down, wearing a gray shirt. His facial expressions transition from calm and relaxed to a gentle smile and laughter, suggesting a pleasant or humorous thought. His eyes are mostly closed, then open slightly, conveying a sense of contentment and ease. The lighting is bright and even, creating a soft focus on his face against a white background. The camera remains steady, focusing on his profile, capturing his subtle emotional shifts with clarity.",
            "The video features a man lying on his back, wearing a gray shirt. Initially, he gazes upward with a neutral, relaxed expression, his curly hair framing his face. As the video progresses, he begins to smile softly, revealing a sense of contentment or amusement. His eyes appear to sparkle with gentle engagement, emphasizing his relaxed demeanor. The lighting is bright and even, highlighting his features against a stark, white background. The camera maintains a close-up, side profile view throughout, subtly shifting focus to capture different angles of his expression and the gradual transformation of his emotions.",
            "The video features a man lying on his back, wearing a gray shirt. Initially, he gazes upward with a neutral, relaxed expression, his curly hair framing his face. As the video progresses, he begins to smile softly, revealing a sense of contentment or amusement. His eyes appear to sparkle with gentle engagement, emphasizing his relaxed demeanor. The lighting is bright and even, highlighting his features against a stark, white background. The camera maintains a close-up, side profile view throughout, subtly shifting focus to capture different angles of his expression and the gradual transformation of his emotions.",
            "The video shows a man lying on his back with curly hair and a beard stubble, wearing a gray T-shirt. His facial expression is neutral and calm, as he gazes upward. The lighting is bright and evenly distributed, suggesting a controlled studio setting. The camera is focused on his face and profile, capturing subtle movements of his eyes and slight shifts in his gaze, maintaining a consistent angle throughout. The background is a plain white, emphasizing the subject's facial features and expression.",
            "The video shows a man lying on his back with curly hair and a beard stubble, wearing a gray T-shirt. His facial expression is neutral and calm, as he gazes upward. The lighting is bright and evenly distributed, suggesting a controlled studio setting. The camera is focused on his face and profile, capturing subtle movements of his eyes and slight shifts in his gaze, maintaining a consistent angle throughout. The background is a plain white, emphasizing the subject's facial features and expression.",
            "In the video, a man with curly hair and a slight beard lies on his back, wearing a gray shirt. The lighting is bright and even, creating a neutral background. The man initially gazes upwards with a calm expression, then his eyes gradually shift as he looks around, reflecting curiosity or contemplation. The camera captures him from a side view, slowly transitioning to a slightly elevated angle. His head tilts progressively downward, suggesting a transition into deeper thought or relaxation. Throughout, the man's demeanor remains calm, with no evident verbal communication, maintaining a serene atmosphere.",
            "In the video, a man with curly hair and a slight beard lies on his back, wearing a gray shirt. The lighting is bright and even, creating a neutral background. The man initially gazes upwards with a calm expression, then his eyes gradually shift as he looks around, reflecting curiosity or contemplation. The camera captures him from a side view, slowly transitioning to a slightly elevated angle. His head tilts progressively downward, suggesting a transition into deeper thought or relaxation. Throughout, the man's demeanor remains calm, with no evident verbal communication, maintaining a serene atmosphere.",
            "The video features a young man with curly hair and a short beard, wearing a grey shirt against a plain white background. He appears to be deeply engaged in conversation, with facial expressions transitioning from inquisitive and slightly puzzled to more intense and assertive. His eyebrows are raised, and his eyes convey focus as he speaks. The lighting is bright and even, highlighting his facial features. The video employs a stationary camera with a close-up shot that remains steady, emphasizing the man's expressions and emotions throughout his silent yet expressive dialogue.",
            "The video features a young man with curly hair and a short beard, wearing a grey shirt against a plain white background. He appears to be deeply engaged in conversation, with facial expressions transitioning from inquisitive and slightly puzzled to more intense and assertive. His eyebrows are raised, and his eyes convey focus as he speaks. The lighting is bright and even, highlighting his facial features. The video employs a stationary camera with a close-up shot that remains steady, emphasizing the man's expressions and emotions throughout his silent yet expressive dialogue.",
            "The video features a young man with curly hair and a subtle stubble, wearing a gray shirt, and is shot against a white background. His expressions transition from curiosity to slight confusion and contemplation, as he shifts his gaze and slightly furrows his brows. The lighting is bright and even, highlighting his features clearly. The camera remains mostly steady, capturing his facial expressions in close-up, suggesting an introspective or questioning moment as he is not speaking but seems to be processing something internally.",
            "The video features a young man with curly hair and a subtle stubble, wearing a gray shirt, and is shot against a white background. His expressions transition from curiosity to slight confusion and contemplation, as he shifts his gaze and slightly furrows his brows. The lighting is bright and even, highlighting his features clearly. The camera remains mostly steady, capturing his facial expressions in close-up, suggesting an introspective or questioning moment as he is not speaking but seems to be processing something internally.",
            "The video features a man in a gray T-shirt with curly hair, who appears to be experiencing discomfort or bracing himself, as indicated by his eyes tightly shut and facial muscles tensed. Throughout the sequence, he remains in a side view and his expressions remain consistent, suggesting a moment of pain or intense concentration. The lighting is bright and even, focusing on his face against a white background. The camera maintains close-up shots of his face, capturing subtle changes in his expression and the slight movements of his head.",
            "The video features a man in a gray T-shirt with curly hair, who appears to be experiencing discomfort or bracing himself, as indicated by his eyes tightly shut and facial muscles tensed. Throughout the sequence, he remains in a side view and his expressions remain consistent, suggesting a moment of pain or intense concentration. The lighting is bright and even, focusing on his face against a white background. The camera maintains close-up shots of his face, capturing subtle changes in his expression and the slight movements of his head.",
            "The video depicts a man with curly hair and a short beard, wearing a gray T-shirt, in a neutral setting with bright, even lighting. His facial expressions change from a relaxed state to a series of intense grimaces, eyes tightly closed, as if responding to a strong sensation or pressure. The close-up shots focus on his face, capturing subtle emotional shifts and physical reactions. There is no apparent movement or dialogue, emphasizing the man's expressive responses, which create a slightly dramatic and contemplative atmosphere. The camera remains steady, maintaining a consistent focus on his face throughout the video.",
            "The video depicts a man with curly hair and a short beard, wearing a gray T-shirt, in a neutral setting with bright, even lighting. His facial expressions change from a relaxed state to a series of intense grimaces, eyes tightly closed, as if responding to a strong sensation or pressure. The close-up shots focus on his face, capturing subtle emotional shifts and physical reactions. There is no apparent movement or dialogue, emphasizing the man's expressive responses, which create a slightly dramatic and contemplative atmosphere. The camera remains steady, maintaining a consistent focus on his face throughout the video.",
            "The video features a woman lying on her back, wearing a brown top, with her hair tied back. Her expressions shift subtly throughout, beginning with a calm, relaxed demeanor and gradually exhibiting a range of emotions from reflective to slightly perplexed, with her lips moving as if speaking or mouthing words. The lighting is bright and even, creating a soft, serene atmosphere. The camera captures her from a side view initially and gradually shifts to a more frontal angle, emphasizing her facial expressions and capturing the nuanced changes in emotion.",
            "The video features a woman lying on her back, wearing a brown top, with her hair tied back. Her expressions shift subtly throughout, beginning with a calm, relaxed demeanor and gradually exhibiting a range of emotions from reflective to slightly perplexed, with her lips moving as if speaking or mouthing words. The lighting is bright and even, creating a soft, serene atmosphere. The camera captures her from a side view initially and gradually shifts to a more frontal angle, emphasizing her facial expressions and capturing the nuanced changes in emotion.",
            "The video features a woman with light blonde hair tied up, wearing a taupe shirt, and initially lying down in a brightly lit, white background setting. Her expressions transition from concentration with a slightly open mouth, to a more relaxed demeanor with a gentle, thoughtful smile. Her eyes convey curiosity and attentiveness, and she occasionally appears to be speaking softly. The video captures her from a side profile to a more direct angle gradually, with smooth, steady camera movements. The lighting remains consistent, highlighting her features and creating a serene and focused atmosphere.",
            "The video features a woman with light blonde hair tied up, wearing a taupe shirt, and initially lying down in a brightly lit, white background setting. Her expressions transition from concentration with a slightly open mouth, to a more relaxed demeanor with a gentle, thoughtful smile. Her eyes convey curiosity and attentiveness, and she occasionally appears to be speaking softly. The video captures her from a side profile to a more direct angle gradually, with smooth, steady camera movements. The lighting remains consistent, highlighting her features and creating a serene and focused atmosphere.",
            "The video features a woman with light hair and a neutral expression, relaxing as she lies on a supportive device for her neck. She wears a brown top, and the scene is brightly lit with a soft, even light. Her eyes are calmly open, suggesting a state of relaxation or meditation. The camera gradually transitions from a close-up of her side profile to a more front-facing angle, highlighting her serene demeanor. Her steady breathing and peaceful gaze indicate a state of comfort, and the minimalist white background amplifies the tranquil atmosphere.",
            "The video features a woman with light hair and a neutral expression, relaxing as she lies on a supportive device for her neck. She wears a brown top, and the scene is brightly lit with a soft, even light. Her eyes are calmly open, suggesting a state of relaxation or meditation. The camera gradually transitions from a close-up of her side profile to a more front-facing angle, highlighting her serene demeanor. Her steady breathing and peaceful gaze indicate a state of comfort, and the minimalist white background amplifies the tranquil atmosphere.",
            "The video showcases a woman with a calm expression lying on her back, wearing a brown top, against a white background. Her eyes are closed initially, showing relaxation, then they open slightly as the camera gradually moves from a side view to a front-facing perspective. The lighting is bright and evenly distributed, creating a soft and serene ambiance. She appears contemplative and relaxed, with subtle movements as she transitions from lying back to a more engaged posture by leaning forward. Her expressions remain neutral, indicating a peaceful state throughout the video. The camera movement is smooth, enhancing the tranquil atmosphere.",
            "The video showcases a woman with a calm expression lying on her back, wearing a brown top, against a white background. Her eyes are closed initially, showing relaxation, then they open slightly as the camera gradually moves from a side view to a front-facing perspective. The lighting is bright and evenly distributed, creating a soft and serene ambiance. She appears contemplative and relaxed, with subtle movements as she transitions from lying back to a more engaged posture by leaning forward. Her expressions remain neutral, indicating a peaceful state throughout the video. The camera movement is smooth, enhancing the tranquil atmosphere.",
            "In a brightly lit setting, a woman wearing a brown top appears engaged and expressive as if she's speaking directly to the camera; her facial expressions shift from neutral to concerned, with slight changes around her eyes and mouth, indicating a progression in her speech or narrative throughout the video. The close-up shots maintain a static composition, focusing on her emotions and expressions, while her consistent eye contact and subtle head movements suggest an interactive or explanatory tone.",
            "In a brightly lit setting, a woman wearing a brown top appears engaged and expressive as if she's speaking directly to the camera; her facial expressions shift from neutral to concerned, with slight changes around her eyes and mouth, indicating a progression in her speech or narrative throughout the video. The close-up shots maintain a static composition, focusing on her emotions and expressions, while her consistent eye contact and subtle head movements suggest an interactive or explanatory tone.",
            "The video features a woman with a calm demeanor, wearing a brown shirt and a delicate necklace, set against a stark white background that creates a soft, even lighting. Her expressions transition subtly from thoughtful to slightly concerned as she speaks, highlighting her expressive eyes and facial movements. The camera remains steady, focusing closely on her face to capture her nuanced expressions and the reflective quality of her gaze. Her head tilts slightly at times, adding a dynamic aspect to her steady yet expressive conversation, which seems introspective or explanatory in nature.",
            "The video features a woman with a calm demeanor, wearing a brown shirt and a delicate necklace, set against a stark white background that creates a soft, even lighting. Her expressions transition subtly from thoughtful to slightly concerned as she speaks, highlighting her expressive eyes and facial movements. The camera remains steady, focusing closely on her face to capture her nuanced expressions and the reflective quality of her gaze. Her head tilts slightly at times, adding a dynamic aspect to her steady yet expressive conversation, which seems introspective or explanatory in nature.",
            "The video features a woman with light hair tied back, wearing a brown shirt. Her facial expressions progress from neutral to subtly puckered lips, indicating a thought-provoking or contemplative mood. Her eyes appear focused, and there's minimal movement, suggesting she might be listening intently or reacting to something unseen. The lighting is bright and even, highlighting her features against a plain white background. The camera remains steady, focusing closely on her face, capturing her subtle emotional changes.",
            "The video features a woman with light hair tied back, wearing a brown shirt. Her facial expressions progress from neutral to subtly puckered lips, indicating a thought-provoking or contemplative mood. Her eyes appear focused, and there's minimal movement, suggesting she might be listening intently or reacting to something unseen. The lighting is bright and even, highlighting her features against a plain white background. The camera remains steady, focusing closely on her face, capturing her subtle emotional changes.",
            "The video appears to feature a woman with light hair tied back, wearing a brown shirt against a white background, captured in a series of close-up frames that display her neutral expression gradually shifting to a subtle frown. Her lips are softly pressed, and her gaze fluctuates as if she is contemplating or reacting to something unseen. The lighting is bright, creating a soft yet shadowless environment that highlights her facial features. The camera remains steady and focused on her face, emphasizing her changing emotions and expressions without any apparent movement.",
            "The video appears to feature a woman with light hair tied back, wearing a brown shirt against a white background, captured in a series of close-up frames that display her neutral expression gradually shifting to a subtle frown. Her lips are softly pressed, and her gaze fluctuates as if she is contemplating or reacting to something unseen. The lighting is bright, creating a soft yet shadowless environment that highlights her facial features. The camera remains steady and focused on her face, emphasizing her changing emotions and expressions without any apparent movement.",
            "The video depicts a woman wearing a brown shirt with her hair tied back, displaying a series of expressions that convey dissatisfaction or contemplation. Her facial expressions remain consistent, with a mild frown and pressed lips as she looks in various directions. The lighting is bright, casting an even glow over her features, which suggests a studio or controlled environment. The camera captures close-up shots from different angles, subtly shifting to emphasize her expressions. Throughout the video, she remains composed, not speaking, allowing her body language to communicate her mood.",
            "The video depicts a woman wearing a brown shirt with her hair tied back, displaying a series of expressions that convey dissatisfaction or contemplation. Her facial expressions remain consistent, with a mild frown and pressed lips as she looks in various directions. The lighting is bright, casting an even glow over her features, which suggests a studio or controlled environment. The camera captures close-up shots from different angles, subtly shifting to emphasize her expressions. Throughout the video, she remains composed, not speaking, allowing her body language to communicate her mood.",
        ],
        lora_name="I2V5B_resum_blendnorm_i22600_webvid",
        # lora_name=None,
        degradation=0.5,
        num_inference_steps=50,
        noise_downtemp_interp="blend_norm",
        subfolder="BATTERY_TEST/VPS_SAMPLES_CUSTOMPROMPT/I2V5B_resum_blendnorm_i22600_webvid",
    )

if False:
    #DL3DV Testing

    r._pterm_cd("/root/CleanCode/Github/cogvideox-factory")

    import ryan_infer

    # lora_name = 'T2V2B_RDeg_i30000'
    lora_name = "I2V5B_final_i30000"
    lora_name = 'T2V5B_blendnorm_i16800_envato'
    # lora_name = 'T2V5B_blendnorm_i16800_envato'

    infer_degrad = 0.5
    steps = 50
    # steps = 30
    interp = "blend_norm"

    settings = f"degrad={infer_degrad}_steps={steps}_interp={interp}"
    subfolder = f"CLEAN_OUTPUTS/WILDCARD_TESTS/BATTERIES/{lora_name}/{settings}"

    sample_path = [
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/twine_duke.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/repel_chump.pkl",
                    "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ahead_job.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/agile_lent.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/agile_wing.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ajar_payer.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/alive_smog.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/amuse_chop.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/argue_life.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/bless_life.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/blog_voice.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/blunt_clay.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/blunt_swab.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/busy_proof.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/carve_stem.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/chomp_shop.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/clump_grub.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/agile_train.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ahead_shred.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/alien_wagon.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/arise_clear.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/balmy_fetch.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/balmy_rerun.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/balmy_smash.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/bleak_skier.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/bless_banjo.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/brisk_stump.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/ajar_doll.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/brisk_rug.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/ajar_clasp.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/bless_scan.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/both_cramp.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/clap_patch.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/amend_shred.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/clink_grief.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/droop_fever.pkl",
            "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples_BlendNoise_Norm_30FPS/elude_barge.pkl",
                    "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/twine_duke.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/agile_five.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/neat_raven.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/wiry_quota.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/snort_denim.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/brisk_carry.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/wipe_doll.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/ahead_vegan.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/thud_wagon.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/lend_aloe.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/blog_ride.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/rich_card.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/thud_pond.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/arise_gift.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/cozy_disk.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/savor_dot.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/polar_reset.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/shout_train.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/elude_jelly.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/relax_crepe.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/dusk_blade.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/enter_void.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/crisp_state.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/cozy_stir.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/clap_chip.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/yelp_elm.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/heap_grave.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/ahead_shell.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/sleek_puppy.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/harm_flip.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/dice_olive.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/plead_video.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/ritzy_plank.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/zoom_grill.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/arise_cache.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/fray_tart.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/speak_deaf.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/repel_think.pkl",
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/repel_chump.pkl",
    ]

    sample_path = shuffled(sample_path)

    prompt = None

    ryan_infer.main(
        sample_path=sample_path,
        lora_name=lora_name,
        degradation=infer_degrad,
        prompt=prompt,
        num_inference_steps=steps,
        noise_downtemp_interp=interp,
        subfolder=subfolder,
    )


if False:
    #PROMPT BATTERIES

    r._pterm_cd("/root/CleanCode/Github/cogvideox-factory")

    import ryan_infer

    # lora_name = 'T2V2B_RDeg_i30000'
    lora_name = "I2V5B_final_i30000"
    #lora_name = 'T2V5B_blendnorm_i11600_envato'

    infer_degrad = 0.5
    steps = 50
    #steps = 30
    interp = "blend_norm"
    #interp = "nearest"

    settings = f"degrad={infer_degrad}_steps={steps}_interp={interp}"
    subfolder = f"CLEAN_OUTPUTS/DL3DV_TESTS/IMAGE_BATTERIES/{lora_name}/{settings}"

    sample_path = [
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/enter_void.pkl",
    ]

    sample_path = shuffled(sample_path)

    image = [
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.44.43 AM.jpg",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.41.54 AM.jpg",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.41.33 AM.jpg",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.40.20 AM.jpg",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.40.09 AM.jpg",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.39.22 AM.jpg",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.39.11 AM.jpg",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.38.55 AM.jpg",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.38.29 AM.jpg",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.36.13 AM.jpg",
    ]


    prompt = [
        """A young woman, dressed in denim and holding a coffee mug, is seated cross-legged on a chair placed on a covered table in front of a "Gameday Spirit Fanstore" sign, set against a brick wall. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """Three strawberry-topped cheesecakes are displayed on a covered table in front of the "Gameday Spirit Fanstore" sign, against a backdrop of a brick wall. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """A single strawberry-topped cheesecake sits on the covered table, prominently in front of the "Gameday Spirit Fanstore" sign and brick wall. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """A fluffy, young Maine Coon kitten with striking markings and bright blue eyes is seated on the covered table in front of the fan store sign and brick wall. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """A Pomeranian dog and a Maine Coon kitten are sitting side by side on the covered table, with the "Gameday Spirit Fanstore" sign and brick wall in the background. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """A fluffy Pomeranian dog is sitting alone on the covered table, in front of the fan store sign and the brick wall. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """The Pomeranian dog is positioned centrally on the table, with its fluffy fur making it appear cute and content in front of the fan store sign. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """A French Bulldog puppy with a curious expression sits on the covered table, positioned in front of the "Gameday Spirit Fanstore" sign. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """The French Bulldog puppy sits on the table, looking directly at the camera, with the "Gameday Spirit Fanstore" sign visible in the background. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """The covered table stands alone in front of the "Gameday Spirit Fanstore" sign against a brick wall, with no objects or animals on it. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
    ]

    prompt, image = sync_shuffled(prompt, image)
    prompt=list(prompt)
    image=list(image)

    ryan_infer.main(
        sample_path=sample_path,
        lora_name=lora_name,
        degradation=infer_degrad,
        prompt=prompt,
        num_inference_steps=steps,
        noise_downtemp_interp=interp,
        subfolder=subfolder,
        image=image,
    )

if False:
    #OBJECT INSERTION
    r._pterm_cd("/root/CleanCode/Github/cogvideox-factory")

    import ryan_infer

    # lora_name = 'T2V2B_RDeg_i30000'
    lora_name = "I2V5B_final_i30000"
    #lora_name = 'T2V5B_blendnorm_i11600_envato'

    infer_degrad = 0.5
    steps = 50
    #steps = 30
    interp = "blend_norm"
    #interp = "nearest"

    settings = f"degrad={infer_degrad}_steps={steps}_interp={interp}"
    subfolder = f"CLEAN_OUTPUTS/DL3DV_TESTS/IMAGE_BATTERIES/{lora_name}/{settings}"

    sample_path = [
        "/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_DL3DV_Samples/enter_void.pkl",
    ]

    sample_path = shuffled(sample_path)

    image = [
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.44.43 AM.png",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.41.54 AM.png",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.41.33 AM.png",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.40.20 AM.png",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.40.09 AM.png",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.39.22 AM.png",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.39.11 AM.png",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.38.55 AM.png",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.38.29 AM.png",
        "/root/CleanCode/Github/cogvideox-factory/datasets/new_first_frames/enter_void/Screenshot 2024-11-06 at 4.36.13 AM.png",
    ]


    prompt = [
        """A young woman, dressed in denim and holding a coffee mug, is seated cross-legged on a chair placed on a covered table in front of a "Gameday Spirit Fanstore" sign, set against a brick wall. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """Three strawberry-topped cheesecakes are displayed on a covered table in front of the "Gameday Spirit Fanstore" sign, against a backdrop of a brick wall. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """A single strawberry-topped cheesecake sits on the covered table, prominently in front of the "Gameday Spirit Fanstore" sign and brick wall. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """A fluffy, young Maine Coon kitten with striking markings and bright blue eyes is seated on the covered table in front of the fan store sign and brick wall. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """A Pomeranian dog and a Maine Coon kitten are sitting side by side on the covered table, with the "Gameday Spirit Fanstore" sign and brick wall in the background. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """A fluffy Pomeranian dog is sitting alone on the covered table, in front of the fan store sign and the brick wall. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """The Pomeranian dog is positioned centrally on the table, with its fluffy fur making it appear cute and content in front of the fan store sign. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """A French Bulldog puppy with a curious expression sits on the covered table, positioned in front of the "Gameday Spirit Fanstore" sign. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """The French Bulldog puppy sits on the table, looking directly at the camera, with the "Gameday Spirit Fanstore" sign visible in the background. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
        """The covered table stands alone in front of the "Gameday Spirit Fanstore" sign against a brick wall, with no objects or animals on it. The video shows a series of scenes in a sports arena, starting with a reflective black cabinet and a gray cloth-covered table, with a person's reflection visible. The scene transitions to a quiet, empty space with a 'Fighting Illini' sign, a draped table, and a brown suitcase. Next, a 'GAMER DAY' sign is seen above a cabinet with a gray cloth, with a suitcase and a black bag nearby. A large blue cabinet with a 'FAN TILIN' sign appears in a sports facility, followed by a similar cabinet with a 'GAMEDAY' sign, a brown suitcase, and a black bag. Finally, a 'GAMER DAY' banner is displayed above a cabinet with a gray cloth, with a suitcase and a black bag on the floor.""",
    ]

    prompt, image = sync_shuffled(prompt, image)
    prompt=list(prompt)
    image=list(image)

    ryan_infer.main(
        sample_path=sample_path,
        lora_name=lora_name,
        degradation=infer_degrad,
        prompt=prompt,
        num_inference_steps=steps,
        noise_downtemp_interp=interp,
        subfolder=subfolder,
        image=image,
    )
