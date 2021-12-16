import glob
import os

def load_time(data_dir, readFile=False, start=None, end=None, isForce=False):
    times = []
    if isForce:
        return [i/(end-start-1) for i in range(0, end-start)] 
    if not readFile:
        image_paths = glob.glob(os.path.join(data_dir, 'images/*'))
        for i in range(len(image_paths)):
            time = i/(len(image_paths) - 1) 
            times.append(time)
        
    else:
        raise NotImplementedError("Read file not implemented")

    return times