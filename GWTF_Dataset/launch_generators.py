from rp import *
command = f'{sys.executable} -m rp call exec_ipynb ---notebook_path "/efs/users/jordanlin/public/ryan/CleanCode/Datasets/Envato/Noisewarp/generator.ipynb"'

yaml = tmuxp_create_session_yaml([[command] * get_num_gpus()]*2, session_name="Noiser")
tmuxp_launch_session_from_yaml(yaml)
