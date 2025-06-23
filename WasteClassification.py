#Importing neccessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')


# In[4]:


train_path="Dataset\TRAIN"
test_path="Dataset\TEST"


# In[5]:


#Importing libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.utils import plot_model
from glob import glob


# In[6]:


#Visualization
from cv2 import cvtColor
x_data=[]
y_data=[]
for category in glob(train_path+'/*'):
    for file in tqdm(glob(category+'/*')):
        img_array=cv2.imread(file)
        img_array=cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        x_data.append(img_array)
        y_data.append(category.split('/')[-1])
data=pd.DataFrame({'image':x_data,'label':y_data})


# In[7]:


data.shape


# In[8]:


colors=['#a0d157','#c48bb8']
plt.pie(data.label.value_counts(),labels=['Organic','Recyclable'],autopct='%0.2f%%',colors=colors,startangle=90,explode=[0.05,0.05])
plt.show()


# In[9]:


plt.figure(figsize=(20,15))
for i in range(9):
    plt.subplot(4,3,(i%12)+1)
    index=np.random.randint(15000)
    plt.title('This is of {0}'.format(data.label[index]))
    plt.imshow(data.image[index])
    plt.tight_layout()


# CNN
# 

# In[10]:


model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",optimizer = "adam",metrics=["accuracy"])
batch_size=256


# In[11]:


model.summary()


# In[12]:


train_datagen=ImageDataGenerator(rescale=1./255)


# In[13]:


test_datagen=ImageDataGenerator(rescale=1./255)


# In[14]:


train_generator=train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical")

test_generator=test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical")


# In[15]:


hist=model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)


# In[16]:


plt.figure(figsize=(10,6))
plt.plot(hist.history['accuracy'],label='Train Accuracy')
plt.plot(hist.history['val_accuracy'],label='Validation Accuracy')
plt.show()


# In[17]:


plt.figure(figsize=(10,6))
plt.plot(hist.history['loss'], label ='Train loss')
plt.plot(hist.history['val_loss'], label = 'Validation loss')
plt.legend()
plt.show()


# In[18]:


def predict_func(img):
    plt.figure(figsize=(6,4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224,3])
    result = np.argmax(model.predict(img))
    if result == 0: 
        print('This image shows recyclable waste')
    elif result ==1: 
        print('This image shows organic waste')


# In[19]:


test_img = cv2.imread("C:/Users/harsh/OneDrive/Desktop/WasteClassification/Dataset/TEST/O/O_12568.jpg")
predict_func(test_img)


# In[29]:


test_img = cv2.imread("C:/Users/harsh/OneDrive/Desktop/WasteClassification/Dataset/TRAIN/R/R_9640.jpg")
predict_func(test_img)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script wasteclassification.ipynb')


# In[ ]:




