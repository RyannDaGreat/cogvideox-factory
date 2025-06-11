from rp import *
with SetCurrentDirectoryTemporarily("/efs/users/jordanlin/public/ryan/CleanCode/Datasets/Envato/Noisewarp"):
    processed_root = "/efs/users/jordanlin/public/ryan/CleanCode/Datasets/Envato/Noisewarp/480P-4x81"
    sample_folders = path_join(
        processed_root,
        file_cache_call(
            f".listdir_{get_folder_name(processed_root)}.lines",
            os.listdir,
            processed_root,
        ),
        show_progress=True,
    )
   
    text_files = path_join(sample_folders,'prompt.txt',show_progress=True)
    video_files=path_join(sample_folders,'video.mp4',show_progress=True)
    noise_files=path_join(sample_folders,'noisewarp/noises.npy',show_progress=True)
    
    prompts = file_cache_call(
        f".prompts_{get_folder_name(processed_root)}_{len(sample_folders)}x.lines",
        load_text_files,
        text_files,
        show_progress=True,
    )
    
    save_file_lines(prompts,'prompts.txt')    
    save_file_lines(get_relative_paths(noise_files),'noises.txt')    
    save_file_lines(get_relative_paths(video_files),'videos.txt')    
