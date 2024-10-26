from rp import *
import torch
from diffusers import CogVideoXImageToVideoPipeline
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

lora_paths = dict(
    x5b_RDeg_i9800         = '/root/CleanCode/Github/CogVideo/finetune/cogvideox5b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate/checkpoint-9800/saved_weights_copy/pytorch_lora_weights.safetensors',
    x5b_0Deg_L512_ND_i1200 = '/root/CleanCode/Github/CogVideo/finetune/cogvideox5b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate-LORA512-0Degrad/checkpoint-1200/saved_weights_copy/pytorch_lora_weights.safetensors',
    x2b_RDeg_i30000        = '/root/CleanCode/Github/CogVideo/finetune/cogvideox2b-lora-single-node-delegator-noisewarp-Oct16-RandomDegradation-LargerBatchSize-SmallLearnRate/checkpoint-30000/saved_weights_copy/pytorch_lora_weights.safetensors',
    x5b_RDeg_L2048_i4800   = '/root/CleanCode/Github/CogVideo/finetune/cogvideox5b-lora-noisewarp-Oct23-LORA2048-RandDegrad-BlendNoiseWithoutNorm/checkpoint-4800/saved_weights_copy/pytorch_lora_weights.safetensors',
    # ...
    x5b_i2v_webvid_i2600   = '/root/CleanCode/Github/cogvideox-factory/outputs/models/cogx-lora-i2v__degrad=0,1__downtemp=nearest__lr=1e-4__2024-10-25T14-52-57-0400/checkpoint-2600/pytorch_lora_weights.safetensors' #Oct26, 3:45AM
)
#To get the trained LoRA paths:
#     >>> lora_paths=glob.glob('/root/CleanCode/Github/CogVideo/finetune/*/*/saved_weights_copy/pytorch_lora_weights.safetensors') #For Old Training Codebase (T2V)
#     >>> lora_paths=glob.glob('/root/CleanCode/Github/cogvideox-factory/outputs/models/*/*/*.safetensors')                        #For New Training Codebase (I2V)
#     ... print(line_join(sorted([max(x, key=by_number) for x in cluster_by_key(lora_paths, lambda x: get_path_parent(x, 3))], key=date_modified)))
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


def get_pipe(pipe_name="T2V5B", lora_name=None, device=None):
    pipe_id = pipe_ids[pipe_name]

    print(f"LOADING PIPE WITH device={device} pipe_id={pipe_id} lora_name={lora_name}")

    is_i2v = "I2V" in pipe_name  # This is a convention I'm using right now
    
    PipeClass = CogVideoXImageToVideoPipeline if is_i2v else CogVideoXPipeline
    pipe = PipeClass.from_pretrained(pipe_ids[pipe_name], torch_dtype=torch.bfloat16)

    pipe.pipe_name = pipe_name

    if lora_name is not None:
        lora_path = lora_paths[lora_name]
        assert file_exists(lora_path)
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

    sample_gif_path = sample_path+'.gif'

    #prompt=sample.instance_prompt
    downtemp_noise = ryan_dataset.downtemp_noise(
        sample_noise,
        noise_downtemp_interp=noise_downtemp_interp,
    )
    downtemp_noise = downtemp_noise[None]
    downtemp_noise = nw.mix_new_noise(downtemp_noise, degradation)

    assert downtemp_noise.shape == (B, F, C, H, W)

    if image is None            : sample_image = rp.as_pil_image(rp.as_numpy_image(sample_video[0].float()/2+.5))
    elif isinstance(image, str) : sample_image = rp.as_pil_image(rp.as_rgb_image(rp.load_image(image)))
    else                        : sample_image = rp.as_pil_image(rp.as_rgb_image(image))

    metadata = gather_vars('sample_path degradation downtemp_noise sample_gif_path sample_video sample_noise noise_downtemp_interp')
    settings = gather_vars('num_inference_steps')

    if noise  is None: noise  = downtemp_noise
    if video  is None: video  = sample_video
    if image  is None: image  = sample_image
    if prompt is None: prompt = sample_prompt

    assert noise.shape == (B, F, C, H, W)

    return gather_vars('prompt noise image video metadata settings')

def get_output_path(pipe, cartridge, subfolder:str, output_root:str="infer_outputs"):

    time = millis()

    def dict_to_name(d=None, **kwargs):
        if d is None:
            d = {}
        d.update(kwargs)
        return ",".join("=".join(map(str, [key, value])) for key, value in d.items())

    output_name = (
        dict_to_name(
            t=time,
            pipe=pipe.pipe_name,
            lora=pipe.lora_name,
            steps    =               cartridge.settings.num_inference_steps,
            degrad   =               cartridge.metadata.degradation,
            downtemp =               cartridge.metadata.noise_downtemp_interp,
            samp     = get_file_name(cartridge.metadata.sample_path, False),
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

def run_pipe(pipe, cartridge, subfolder="first_subfolder"):
    output_mp4_path = get_output_path(pipe, cartridge, subfolder)
    
    video = pipe(
        prompt=cartridge.prompt,
        **(dict(image=cartridge.image) if pipe.is_i2v else {}),
        num_inference_steps=cartridge.settings.num_inference_steps,
        latents=cartridge.noise.to(pipe.device),

        # FYI, SOME OTHER DEFAULT VALUES:
        # num_videos_per_prompt=1,
        # num_frames=num_frames,
        # guidance_scale=6,
        # generator=torch.Generator(device=device).manual_seed(42),
    ).frames[0]

    export_to_video(video, output_mp4_path, fps=8)

    sample_gif=load_video(cartridge.metadata.sample_gif_path)
    video=as_numpy_images(video)
    prevideo=horizontally_concatenated_videos(resize_list(video,len(sample_gif)),sample_gif)
    preview_mp4_path=rp.save_video_mp4(prevideo, output_mp4_path+'_preview.mp4',framerate=16)

    return gather_vars('output_mp4_path preview_mp4_path cartridge subfolder')


# #prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
# prompt = "An old house by the lake with wooden plank siding and a thatched roof"
# prompt = "Soaring through deep space"
# prompt = "Swimming by the ruins of the titanic"
# prompt = "A camera flyby of a gigantic ice tower that a princess lives in, zooming in from far away from the castle into her dancing in the window"
# prompt = "A drone flyby of the grand canyon, aerial view"
# prompt = "A bunch of puppies running around a front lawn in a giant courtyard "
# #image = load_image(image=download_url_to_cache("https://media.sciencephoto.com/f0/22/69/89/f0226989-800px-wm.jpg"))


pipes = []
if 0 or not pipes:
    for device in get_all_gpu_ids()[:2]:
        pipes.append(
            get_pipe(
                pipe_name="I2V5B",
                lora_name="x5b_i2v_webvid_i2600",
                device=device,
            )
        )
    
cartridges = []
for _ in pipes:
    cartridges.append(
        load_sample_cartridge(
            num_inference_steps=30,
        ),
    )

run_pipe(pipes[0],cartridges[0])
