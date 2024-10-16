import kagglehub
import os
os.environ['KAGGLE_CONFIG_DIR'] = '.'
os.environ['XDG_CACHE_HOME'] = '.'
os.system("export XDG_CACHE_HOME=/project/pi_rrahimi_umass_edu/vishalg/CS682-Course-Project")
os.system("export KAGGLE_CONFIG_DIR=/project/pi_rrahimi_umass_edu/vishalg/CS682-Course-Project")
os.system('export HF_DATASETS_CACHE="/project/pi_rrahimi_umass_edu/vishalg/hf_cache"')
os.system('export HF_HOME="/project/pi_rrahimi_umass_edu/vishalg/hf_cache"')
kagglehub.config.cache_dir = '/project/pi_rrahimi_umass_edu/vishalg/CS682-Course-Project'
# Download latest version
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")

print("Path to dataset files:", path)