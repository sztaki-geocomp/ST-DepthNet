from __future__ import print_function, division
from tensorflow.keras.layers import Input, Concatenate, ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
import datetime
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time
import psutil

from data_utils import get_data_files, get_data

class STDepthNet(tf.keras.Model):
    def __init__(self, im_width=400, im_height=400, channels=1, lookback=5):
        super(STDepthNet, self).__init__()
        
        #Loss weigths
        self.ssim_loss_weight = 0.70
        self.l1_loss_weight = 1.4
        self.edge_loss_weight = 1.5
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        #Initial paramaters
        self.img_rows = im_height
        self.img_cols = im_width
        self.channels = channels
        self.lookback = lookback
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_seq_shape = (self.lookback + 1,) + self.img_shape   
        self.ssim_loss = 0
        self.l1_loss = 0
        self.depth_s = 0

        disc = 4
        patch = int(self.img_rows / 2**disc)
        self.disc_patch = (patch, patch, 1)
        #Filter number
        self.gf = 32 
        self.df = 32
        
        #Optimizer
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        img_seq_A = Input(shape=self.img_seq_shape)
        img_B = Input(shape=self.img_shape)

        fake_B = self.generator(img_seq_A)
        #If you want to train with a discriminator turn true
        self.discriminator.trainable = False
        valid = self.discriminator([img_seq_A, fake_B])

        self.combined = Model([img_seq_A, img_B], [valid, fake_B])
        
        #Adding the custom loss function to tensorflow
        self.combined.compile(loss=self.call, optimizer=optimizer)
    
    #Build the generator ST-DepthNet network 
    def build_generator(self): 
        #Loading the five input images
        inputs = tf.keras.Input(shape=(5, 400, 400, 1))
        #Indexing out the last one
        input5 = layers.Lambda(lambda x: x[:,4,:,:,:])(inputs)
        
        #Construction of the layers
        x = layers.ConvLSTM2D(8, kernel_size=(3, 3), padding='same', strides = (1, 1), return_sequences=True)(inputs)
        x = layers.Activation('relu')(x)
        x1 = layers.BatchNormalization()(x)

        x = layers.ConvLSTM2D(8, kernel_size=(3, 3), padding='same', strides = (2, 2), return_sequences=True)(x1)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.ConvLSTM2D(32, kernel_size=(3, 3), padding='same', strides = (1, 1), return_sequences=True)(x)
        x = layers.Activation('relu')(x)
        x2 = layers.BatchNormalization()(x)

        x = layers.ConvLSTM2D(32, kernel_size=(3, 3), padding='same', strides = (2, 2), return_sequences=True)(x2)
        x = layers.Activation('relu')(x)
        x3 = layers.BatchNormalization()(x)

        x = layers.ConvLSTM2D(64, kernel_size=(3, 3), padding='same', strides = (2, 2), return_sequences=True)(x3)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.ConvLSTM2D(64, kernel_size=(3, 3), padding='same', strides = (1, 1), return_sequences=False)(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)

        ### [Second half of the network: upsampling inputs] ###
        
        x =  layers.UpSampling2D(size=2)(x)
        x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)
        xx3 = layers.BatchNormalization()(x)

        # Residual 3
        inp3 = layers.ConvLSTM2D(64, kernel_size=(3, 3), padding='same', strides = (1, 1), return_sequences=False)(x3)
        x = layers.Add()([xx3, inp3])

        x =  layers.UpSampling2D(size=2)(x)
        x = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)
        xx2 = layers.BatchNormalization()(x)

        # Residual 2
        inp2 = layers.ConvLSTM2D(32, kernel_size=(3, 3), padding='same', strides = (1, 1), return_sequences=False)(x2)
        x = layers.Add()([xx2, inp2])

        x =  layers.UpSampling2D(size=2)(x)
        x = Conv2D(8, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)
        xx1 = layers.BatchNormalization()(x)

        # Residual 1
        inp1 = layers.ConvLSTM2D(8, kernel_size=(3, 3), padding='same', strides = (1, 1), return_sequences=False)(x1)
        x = layers.Add()([xx1, inp1])
        
        # Add a per-pixel classification layer
        outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
        x = layers.Add()([outputs, input5])
        # Define the model
        model = Model(inputs, x)

        return model
    
    #Build a discriminator network
    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = tf.keras.layers.LeakyReLU(alpha=0.0)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        img_A = Input(shape=self.img_seq_shape)
        img_B = Input(shape=self.img_shape)
        lstm_out = ConvLSTM2D(filters=self.df, kernel_size=4, padding="same")(img_A)
        lstm_out = tf.keras.layers.LeakyReLU(alpha=0.2)(lstm_out)
        combined_imgs = Concatenate(axis=-1)([lstm_out, img_B])
        d1 = d_layer(combined_imgs, self.df)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        return Model([img_A, img_B], validity)
    
    #Custom loss function
    def call(self, target, pred):
        # Edges
        target = tf.convert_to_tensor(tf.cast(target, tf.float32))
        pred = tf.convert_to_tensor(tf.cast(pred, tf.float32))       
        dy_true, dx_true = tf.image.image_gradients(target)
        dy_pred, dx_pred = tf.image.image_gradients(pred)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

        # Depth smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y

        self.depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
            abs(smoothness_y)
        )

        # Structural similarity (SSIM) index
        self.ssim_loss = tf.reduce_mean(
            1 - tf.image.ssim(target,pred,max_val=1.0)
        )
             
        # Point-wise depth
        self.l1_loss = tf.reduce_mean(tf.abs(target - pred))
        loss = (
            (self.ssim_loss_weight * self.ssim_loss)
            + (self.l1_loss_weight * self.l1_loss)
            + (self.edge_loss_weight * self.depth_smoothness_loss)
        )
        return loss
    
    #Train function
    def train(self, epochs, batch_size=1, save_interval=4000, save_file_name="st-depthnet.model"):
        print("TRAINING STARTED:")
        
        saver = 1
        figure, axe = plt.subplots(1, 3, squeeze=False, figsize=(1*10, 1*10))
        for epoch in (range(epochs)):
            #Loads the data for every epoch
            counter = 1
            train_files, test_files = get_data_files()     
            train_gen = get_data(files=train_files, timesteps=5, batch_size=batch_size, im_size=(self.img_rows, self.img_rows))
            start_for_one_total_epoch_time = time.time()
            print("Starting epoch: ", str(epoch))
            #Training loop for one epoch
            while True:
                start_for_epoch_time = time.time()
                #Gets the data for one batch
                try:
                    img_seqs_A, imgs_B = next(train_gen)
                except StopIteration:
                    self.generator.save(save_file_name + "_" +str(epoch) + ".model")
                    break
                #Generates output image
                fake_B = self.generator.predict(img_seqs_A)
                #Saving some images for visualization
                if counter % save_interval == 0:
                    MYDIR = ("./prediction/training_plot/" + save_file_name + "/")
                    CHECK_FOLDER = os.path.isdir(MYDIR)
                    if not CHECK_FOLDER:
                        os.makedirs(MYDIR)
                    axe[0, 0].set_ylabel("Ground thruth")
                    axe[0, 0].imshow(np.squeeze(imgs_B), interpolation="none", cmap='gray')
                    
                    axe[0, 1].set_ylabel("Actual")
                    axe[0, 1].imshow(np.squeeze(img_seqs_A)[4], interpolation="none", cmap='gray')
                    
                    axe[0, 2].set_ylabel("Predicted")
                    axe[0, 2].imshow(np.squeeze(fake_B), interpolation="none", cmap='gray')
                    figure.subplots_adjust(wspace=0.5, hspace=0.5)
                    figure.savefig(MYDIR + str(epoch) + "_epoch_" + str(counter) + "_iteration" + ".png", bbox_inches='tight')
                    saver += 1
                #This part is only relevant for GAN-based training
                valid = np.ones((batch_size,) + self.disc_patch)
                fake = np.zeros((batch_size,) + self.disc_patch)       
                d_loss_real = self.discriminator.train_on_batch([img_seqs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([img_seqs_A, fake_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                valid = np.ones((batch_size,) + self.disc_patch)
                
                #Train
                g_loss = self.combined.train_on_batch([img_seqs_A, imgs_B], [valid, imgs_B])
                #Some information about the training
                print("Iterations done: ", counter, " Iteration execution time: %s seconds" % (time.time() - start_for_epoch_time),
                      " Total execution time for current epoch: %s seconds" % (time.time() - start_for_one_total_epoch_time),
                      " Generator loss: ", g_loss[0]," Ram usage: ", psutil.Process().memory_info().rss / (1024 * 1024))
                counter += 1
        self.generator.save(save_file_name + "_final.model")
