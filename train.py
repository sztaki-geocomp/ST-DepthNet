from model import STDepthNet
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import argparse

#Additional informations are provided in the README file



def getOpt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--image_size", type=int, default=400, help="size of the images")
    parser.add_argument("--save_model", type=str, default="./models/model", help="path to save the model to")
    return parser.parse_args()

with tf.device('/device:GPU:0'):
    #Gets the options from command line
    opt = getOpt()
    #Training on GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False

    sess = tf.Session(config=config)
    set_session(sess)
    
    #Sets the parameters
    epochs = opt.n_epochs
    batch_size = opt.batch_size
    timesteps = 5
    im_width = im_height = opt.image_size
    #Gets the model
    net = STDepthNet(im_height=im_height, im_width=im_width, lookback=timesteps-1)
    #Trains the model
    net.train(epochs=epochs, batch_size=batch_size, save_interval=200, save_file_name=opt.save_model)
