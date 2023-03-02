import glob
import random
import numpy as np
import sys
from PIL import Image
from moviepy.editor import VideoFileClip
import cv2 


#Additional informations are provided in the README file


first = True
counter =  0
#Getting the data from the given video
def get_data_files(split_ratio=1):
    files = []
    video_files = glob.glob("./dataset/train.avi")
    files.extend(video_files)
    total_files = len(files)
    len_train_files = int(total_files * split_ratio)
    return files[:len_train_files], files[len_train_files:]
#Normalize
def normalize_image(img):
    return img / 255.0
#Denormalize
def denormalize(img):
    return (img * 255).astype('uint8')
#Gets xy pairs
def xy_pair(frames, timesteps, frame_mode, im_size):
    global first
    global counter
    np.set_printoptions(threshold=sys.maxsize)
    assert frame_mode in ["all", "unique"], "frame_mode must be either of unique or all"
    rng = range(timesteps, len(frames)-1, timesteps+1) if frame_mode=="unique" else range(timesteps, len(frames)-1)
    def resize_image(np_image):
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        return np.array(Image.fromarray(np_image, mode='L').resize(im_size))
    for i in rng:
        x = frames[i-timesteps: i]
        x = list(map(resize_image, x))
        for index, item in enumerate(x):
            x[index] = np.expand_dims(x[index], axis=2)
        y = np.reshape(resize_image(frames[i]), (im_size[0], im_size[0], 1))
        yield x, y
#Returns the data
def get_data(files, timesteps=5, fps=6, batch_size=32, frame_mode="unique", im_size=(400, 400)):
    global counter
    x_batch = []
    y_batch = []
    file = files
    clip = VideoFileClip(file[0], audio=False)
    frames = list(clip.iter_frames(fps=fps))
    clip.close()
    
    for x, y in xy_pair(frames, timesteps=timesteps, frame_mode=frame_mode, im_size=im_size):
        counter += 1
        x_batch.append(x)
        y_batch.append(y)
        if len(x_batch) >= batch_size:
            yield normalize_image(np.array(x_batch)), normalize_image(np.array(y_batch))
            x_batch = []
            y_batch = []
    return StopIteration
    
