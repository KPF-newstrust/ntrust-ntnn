import glob
import shutil
from pathlib import Path


__home__ = str(Path.home())

def list_files(pattern):
    files = glob.glob(pattern)
    return sorted(files)

def rmdir(path):
    def onerror(fn, path, info):
        print(info)
        sys.exit()

    shutil.rmtree(path, ignore_errors=False, 
        onerror=onerror)

def get_ckpt(model_dir):
    ckpt = list_files(model_dir + '/model.ckpt-*')[0]
    return ckpt[:ckpt.rfind('.')]

