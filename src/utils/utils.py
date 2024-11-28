import os
import random
from sklearn.model_selection import train_test_split
import warnings

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def split_dataset(directory_path, train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15, seed = 42):
    files = []
    for f in os.listdir(directory_path):         
        if f.startswith('.'):
            warning = "Warning: Ignoring hidden files/folders - " + f
            warnings.warn(warning)
            continue
        
        if f.split('.')[0].endswith('_ic'):
            # Ignoring the incorrect files
            continue

        if os.path.isfile(os.path.join(directory_path, f)):
            files.append(os.path.join(directory_path, f))
    
    random.seed(seed)
    random.shuffle(files)

    train_files, temp_files = train_test_split(files, train_size = train_ratio, random_state = seed)
    val_files, test_files = train_test_split(temp_files, test_size = test_ratio / (val_ratio + test_ratio), random_state = seed)

    return {
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }
