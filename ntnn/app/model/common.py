import os, re
from tensorflow.contrib.predictor import from_saved_model

def load_model(model_dir):
    dirs = os.listdir(model_dir)
    dirs = [d for d in dirs if re.match(r'^\d+$', d)]
    dirs.sort()
    if not len(dirs): 
        raise Exception('invalid export dir: %s' % model_dir)
    return from_saved_model(export_dir=os.path.join(model_dir, dirs[-1]))


