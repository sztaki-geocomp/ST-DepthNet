from data_utils import get_data, get_data_files
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageDraw
import time
import os
from tensorflow.python.keras.models import load_model
#Using the model to predict the depth map with GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
sess = tf.Session(config=config)
import argparse

#Additional informations are provided in the README file


#Gets the options from command line
def getOpt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_type", type=int, default=0, help="choosing the type of the plot")
    return parser.parse_args()
#Loads the models, delete second one if not needed
print("Loading model...")
model = load_model("./models/model1.model", compile=False)
model2 = load_model("./models/model2.model", compile=False)
print("Loaded model")
im_width = im_height = 400
delta = 5
#Mask for the images
mask = cv2.imread("./utils/mask_400.png", cv2.IMREAD_GRAYSCALE)
#Plots the images in a row
def row_plot(x, y_true, y_pred, filename):
    figure, axe = plt.subplots(1, 3, squeeze=False, figsize=(1*10, 1*10), constrained_layout = True)

    axe[0, 0].set_ylabel("Actual grayscale image")
    img = np.squeeze(x)[0]
    axe[0, 0].imshow(img[:,:], interpolation="none", cmap='gray')
    
    axe[0, 1].set_ylabel("Ground thruth")
    img = y_true
    img = np.squeeze(y_true)
    axe[0, 1].imshow(np.squeeze(img), interpolation="none", cmap='gray')
    
    axe[0, 2].set_ylabel("Prediction of the model")
    img = y_pred
    img = np.squeeze(y_pred)
    img = img * 255
    img[img < delta] = 0
    img = img / 255
    img[mask == 0] = 0
    axe[0, 2].imshow(np.squeeze(img), interpolation="none", cmap='gray')
    
    figure.subplots_adjust(wspace=0.5, hspace=0.5)
    figure.savefig(filename, bbox_inches='tight')

#Plots the images in a matrix
def matrix_plot(x, y_true, y_pred, y_pred2, filename):
    figure, axe = plt.subplots(2, 2, squeeze=False, figsize=(1*10, 1*10), constrained_layout = True)

    axe[0, 0].set_ylabel("Actual")
    img = np.squeeze(x)[0]
    axe[0, 0].imshow(img[:,:], interpolation="none", cmap='gray')
    
    axe[0, 1].set_ylabel("Ground Truth")
    img = y_true
    axe[0, 1].imshow(np.squeeze(img), interpolation="none", cmap='gray')
    
    axe[1, 0].set_ylabel("Predicted 1")
    img = np.squeeze(y_pred)
    img = img * 255
    img[img < delta] = 0
    img = img / 255
    img[mask == 0] = 0
    axe[1, 0].imshow(img, interpolation="none", cmap='gray')
    
    axe[1, 1].set_ylabel("Predicted 2")
    img = np.squeeze(y_pred2)
    img = img * 255
    img[img < delta] = 0
    img = img / 255
    img[mask == 0] = 0
    axe[1, 1].imshow(img, interpolation="none", cmap='gray')
    
    
    figure.subplots_adjust(wspace=0.2, hspace=0.2)
    figure.savefig(filename, bbox_inches='tight')



#Plots one image
def solo_plot_results(y_true, y_pred, filename):
    tmp = np.squeeze(y_pred)*255
    tmp[tmp < delta] = 0
    tmp[mask == 0] = 0
    cv2.imwrite(filename, tmp)

    





#PLotting function
def visualize(filename, plot_type=0):
    matrix_plot_bool = False
    solot_plot_bool = False
    row_plot_bool = False
    folder_name = ""
    if plot_type == 2:
        matrix_plot_bool= True
        folder_name = "matrix_plot"
    elif plot_type == 1:
        solot_plot_bool = True
        folder_name = "solo_plot"
    elif plot_type == 0: 
        row_plot_bool = True
        folder_name = "row_plot"
    filename = filename + folder_name
    MYDIR = (filename)
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
    train_files, test_files = get_data_files()
    test_gen = get_data(files=train_files, timesteps=timesteps, 
                        batch_size=batch_size, 
                        im_size=(400, 400),
                        fps=6                  
                        )
    if matrix_plot_bool:
        test_gen2 = get_data(files=train_files, timesteps=timesteps, 
                        batch_size=batch_size, 
                        im_size=(400, 400),
                        fps=6                  
                        )
    filename = filename + "/"
    #change the range to the number of images you want to visualize
    for i in range(1225):
        x, y_true = next(test_gen)
        #Measuring the time
        start = time.time()
        y_pred = model.predict(x, batch_size=5)
        end = time.time()
        print("Prediction time of the model: ", end - start)
        if row_plot_bool:
            name = filename
            row_plot(x, y_true, y_pred, filename=name + str(i+1).zfill(6) + ".png")
        if solot_plot_bool:
            name = filename
            solo_plot_results(y_true, y_pred, filename=name + str(i+1).zfill(6) + ".png")
        if matrix_plot_bool:
            x2, y_true2 = next(test_gen2)
            name = filename
            y_pred_2 = model2.predict(x2, batch_size=5)
            matrix_plot(x, y_true, y_pred, y_pred_2, filename=name + str(i+1).zfill(6) + ".png")


batch_size = 1
timesteps = 5
#Gets options from the command line
opt = getOpt()
plot_type = opt.plot_type
#Calling the plotting function
visualize("./prediction/", plot_type=plot_type)
