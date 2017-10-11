
# coding: utf-8

# In[1]:

#from sklearn.datasets import load_files       
#from keras.utils import np_utils
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,Dense
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.models import Sequential
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2      
from extract_bottleneck_features import *
from glob import glob
import argparse

dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]


# In[2]:
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
ResNet50_model = ResNet50(weights='imagenet')
#bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
#train_Resnet50 = bottleneck_features['train']
#valid_Resnet50 = bottleneck_features['valid']
#test_Resnet50 = bottleneck_features['test']


# In[3]:

Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=(1,1,2048)))
Resnet50_model.add(Dense(133, activation='softmax'))
Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')


# In[4]:

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


# In[5]:

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# In[6]:

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


# In[7]:

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


# In[8]:

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# In[9]:

def predict_breed_Resnet50(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


# In[10]:

def predict_breed2(img_path):
    #plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    #plt.show()
    if dog_detector(img_path):
        print ("That's  dog,  Breed : %s" % (predict_breed_Resnet50(img_path)))
    elif face_detector(img_path):
        print ("It's a Human, resembling a %s" % (predict_breed_Resnet50(img_path)))
    else:
        print ("ERROR: No dog or human face detected in %s:" % img_path)


# In[11]:

predict_breed2((args["image"]))


# In[ ]:


