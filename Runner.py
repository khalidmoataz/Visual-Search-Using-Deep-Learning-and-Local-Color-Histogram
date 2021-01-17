from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
import os
import annoy
from annoy import AnnoyIndex
from keras.models import load_model
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt


from Data_preprocessing import *

from Get_Similar_Images import *


img_array,images_features,Rgb_colors = pre_process_data('C:\\Users\\khali\\101_ObjectCategories\\random',VGG_Model)
preds1,new,the_images,Rgb = process_query_image('C:\\Users\\khali\\cam.jpg',images_features,img_array,Rgb_colors,VGG_Model)
nearest_neighbors = get_nearest_neighbor_and_similarity(preds1,15)
results = get_similar_images(nearest_neighbors,the_images,new)
display_results_before_LCH(Rgb,results,nearest_neighbors)
display_results_after_LCH(Rgb,results,nearest_neighbors)
