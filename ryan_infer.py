from rp import *
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video, load_image
from icecream import ic
import rp.git.CommonSource.noise_warp as nw

def dict_to_name(d=None,**kwargs):
	if d is None:
		d={}
	d.update(kwargs)
	return '_'.join('='.join(map(str,[key,value])) for key,value in d.items())

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


if 'get_pipe' not in dir():
	@memoized
	def get_pipe(pipe_name='T2V5B', lora_name=None, device=None):
		pipe_id = pipe_ids[pipe_name]
		lora_path = lora_paths[lora_name]

		print(f"LOADING PIPE WITH device={device} pipe_id={pipe_id} lora_path={lora_path}")

		pipe = CogVideoXPipeline.from_pretrained(pipe_ids[pipe_name],torch_dtype=torch.bfloat16)

		pipe.pipe_name = pipe_name

		if lora_path is not None:
			assert file_exists(lora_path)
			print(end="\tLOADING LORA WEIGHTS...")
			pipe.load_lora_weights(lora_path)
			print('DONE!')

		if device is None:
			device = select_torch_device()

		if device is not None:
			print("\tUSING PIPE DEVICE",device)
			pipe = pipe.to(device)

		#pipe.enable_sequential_cpu_offload()
		pipe.vae.enable_tiling()
		pipe.vae.enable_slicing()

		return pipe

dtype=torch.bfloat16


#https://medium.com/@ChatGLM/open-sourcing-cogvideox-a-step-towards-revolutionizing-video-generation-28fa4812699d
B, F, C, H, W = 1, 13, 16, 60, 90  # The defaults
num_frames=(F-1)*4+1 #https://miro.medium.com/v2/resize:fit:1400/format:webp/0*zxsAG1xks9pFIsoM
#Possible num_frames: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49

latents = torch.randn((B, F, C, H, W), device=device, dtype=dtype)

#prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
prompt = "An old house by the lake with wooden plank siding and a thatched roof"
prompt = "Soaring through deep space"
prompt = "Swimming by the ruins of the titanic"
prompt = "A camera flyby of a gigantic ice tower that a princess lives in, zooming in from far away from the castle into her dancing in the window"
prompt = "A drone flyby of the grand canyon, aerial view"
prompt = "A bunch of puppies running around a front lawn in a giant courtyard "
#image = load_imxage(image=download_url_to_cache("https://media.sciencephoto.com/f0/22/69/89/f0226989-800px-wm.jpg"))

num_inference_steps = 30 #USER INPUT++
num_inference_steps = 50 #USER INPUT++

#FUCK WITH THE LATENTS

# #ORIGINAL TYPES
# for i in range(1,F):
#     #latents[:,i]/=latents[:,i].std(2,keepdim=True) #Normalize them by their channel variances
#     latents[0,i]=latents[0,i-1].roll(10,dims=-2)
#     latents[0,i][:10]=torch.randn_like(latents[0,i][:10]) #Don't loop! We got full 3d attention!

#NON-LOOPING ROLL (BTCHW Form) https://tinyurl.com/2ywlk6w4 for demo

if 1:
	for i in range(1,F):
		#LEFT / RIGHT
		ROLL=5
		latents[:,i,:,:,ROLL:]=latents[:,i-1,:,:,:-ROLL]
		latents[:,i,:,:,:ROLL]=torch.randn_like(latents[:,i,:,:,:ROLL])

if 1:
	for i in range(1,F):
		#UP / DOWN
		ROLL=5
		latents[:,i,:,ROLL:,:]=latents[:,i-1,:,:-ROLL,:]
		latents[:,i,:,:ROLL,:]=torch.randn_like(latents[:,i,:,:ROLL,:])
	
if 1:
	#COMPLETELY FROM SAMPLE: Generate with /root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidSampleGenerator.ipynb
	# sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/plus_pug.pkl'
	sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/amuse_chop.pkl'
	sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/chomp_shop.pkl'
	
	#sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ahead_job.pkl'

	#sample_path = rp.random_element(glob.glob('/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/*.pkl'))

	sample_gif = sample_path+'.gif'
	print(end="LOADING "+sample_path+"...")
	sample=rp.file_to_object(sample_path)
	print("DONE!")

	#prompt=sample.instance_prompt
	latents=resize_list(sample.instance_noise.to(latents.dtype).to(latents.device), latents.shape[1])[None]
	ic(latents.shape)
	

#DEGREDATION
#degrade=0
#degrade=.5
#degrade=.8
degrade=.6
degrade=0
#degrade=1
latents=nw.mix_new_noise(latents,degrade)

#END FUCK WITH LATENTS
	
time=millis()

output_name=dict_to_name(gather_vars('time pipe_name num_frames num_inference_steps degrade F H W'))+'.mp4'
output_folder=make_directory('outputs')
output_path=get_unique_copy_path(path_join(output_folder,output_name))

ic(pipe_name, num_frames, num_inference_steps, output_path, degrade, prompt, device)


video = pipe(
	prompt=prompt,
	#image=image,
	num_videos_per_prompt=1,
	num_inference_steps=num_inference_steps,
	num_frames=num_frames,
	guidance_scale=6,
	generator=torch.Generator(device=device).manual_seed(42),
	latents=latents,
).frames[0]

#Save the video
export_path=path_join(make_directory('OUTPUTS/warp_simple2B_customtestOct23/'+get_file_name(sample_path,False)),output_path)
make_directory(get_path_parent(export_path))
export_to_video(video, export_path, fps=8)

sample_gif=load_video(sample_gif)
video=as_numpy_images(video)
prevideo=horizontally_concatenated_videos(resize_list(video,len(sample_gif)),sample_gif)
mp4_path=rp.save_video_mp4(prevideo,export_path+'preview.mp4',framerate=16)
convert_to_gif_via_ffmpeg(mp4_path,framerate=16)
