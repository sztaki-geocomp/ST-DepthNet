import cv2
import os
from tqdm import tqdm
#Pathes
directory = './depth_image_200ms/'
directory_gt =  './depth_image_1000ms/'
names = []
names_gt = []
file_sample = []
file_gt = []
#Loads the images
for filename in os.listdir(directory):  
    names.append(filename)

names.sort()
print(names)
for name in names:
    file_sample.append(os.path.join(directory, name) + '/')
    file_gt.append(os.path.join(directory_gt, name) + '/')
file_sample.sort()
names_gt.sort()
#Constructiong video
video_name = './video.avi'
height = 400
width = 400 
layers = 400
video = cv2.VideoWriter(video_name, 0, 6, (width,height))
counter = 0
save = 0
filename_counter = 0
#Looping over the images and saving them in the video
for filename in tqdm(file_sample):
    image_folder = filename
    
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    tmp = []
    for image in images:
        if counter >= save:
            video.write(cv2.imread(os.path.join(image_folder, image)))

    for _gt in sorted(os.listdir(file_gt[counter])):
        tmp.append(_gt)
    
    tmp.sort()
    gt = cv2.imread(os.path.join(file_gt[counter] +'/'+ tmp[len(tmp)-1]))
    if counter >= save:
        video.write(gt)
    #Break after N images
    if counter > 8000:
        break
    counter += 1
#Releasing the video
cv2.destroyAllWindows()
video.release()