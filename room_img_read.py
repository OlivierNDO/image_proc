# Configuration
###############################################################################
# Import packages
import numpy as np
import skimage
from PIL import Image
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

# File configuration
config_folder_path = 'D:/indoor_scenes/cloud_trained_models/resnet_220_220_002/'
config_model_file = 'cloud_scene_img_model_sgd_lrs_20191007_0313.hdf5'

# Example image
config_example_img = 'https://ssl.cdn-redfin.com/photo/108/bigphoto/983/1413983_19_0.jpg'

# Define functions / classes
###############################################################################


class RoomPrediction:
    """
    Predict room probabilities from image url or local file path
    Args:
        img : either url or file path to local machine
        img_type: {'url' or 'file'} depending on <img> attribute
        pred_type: type of prediction output {'numeric' or 'display'}
        img_height: height of images used in model training
        img_width: width of images used in model training
        model_object: loaded Keras model
    """
    def __init__(self,
                 img,
                 model_object,
                 img_type = 'url',
                 pred_type = 'numeric',
                 img_height = 220,
                 img_width = 220):
        self.img = img
        self.img_type = img_type
        self.pred_type = pred_type
        self.img_height = img_height
        self.img_width = img_width
        self.model_object = model_object
        

    def read_resize_image(self):
        """Read image (url or file path) and resize"""
        if self.img_type == 'url':
            img_load = Image.open(BytesIO(requests.get(self.img).content))
        elif self.img_type == 'file':
            img_load = tensorflow.keras.preprocessing.image.load_img(self.img)
        else:
            print('Error: Attribute img_type must be "url" or "file"')
        
        resized_img = skimage.transform.resize(np.array(img_load),  (self.img_height, self.img_width))
        return np.expand_dims(np.array(resized_img), axis = 0)
        
    def predict_rooms(self):
         input_image = self.read_resize_image()
         pred_list = list(model_object.predict(input_image)[0])
         if self.pred_type == 'display':
             pred_dict = dict(zip(['Bathroom', 'Bedroom', 'Diningroom', 'Kitchen', 'Livingroom'],
                                  [str(round(p * 100,3)) + "%" for p in pred_list]))
         else:
             pred_dict = dict(zip(['Bathroom', 'Bedroom', 'Diningroom', 'Kitchen', 'Livingroom'],
                                  pred_list))
         return pred_dict
         
     
# Execute on example
###############################################################################      
model_object = keras.models.load_model('{}{}'.format(config_folder_path, config_model_file))
room_predicter = RoomPrediction(model_object = model_object, img = config_example_img, img_type = 'url', pred_type = 'display')
preds = room_predicter.predict_rooms()
print(preds)







