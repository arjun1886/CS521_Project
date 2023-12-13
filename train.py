import os
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model
import keras
from keras.callbacks import ModelCheckpoint
import librosa
from keras.layers import Conv2D, MaxPooling2D
import warnings
warnings.filterwarnings("ignore")


data_path = "data"
data_dir_list = os.listdir(data_path)
print(data_dir_list)
map=[[1 for i in range(2)] for j in range(10)]

for i in range(len(data_dir_list)):
    if data_dir_list[i]=='eight':
       map[i][0]='eight'
       map[i][1]=8
    elif data_dir_list[i]=='five':
       map[i][0]='five'
       map[i][1]=5
    elif data_dir_list[i]=='four':
       map[i][0]='four'
       map[i][1]=4
    elif data_dir_list[i]=='nine':
       map[i][0]='nine'
       map[i][1]=9
    elif data_dir_list[i]=='one':
       map[i][0]='one'
       map[i][1]=1
    elif data_dir_list[i]=='seven':
       map[i][0]='seven'
       map[i][1]=7
    elif data_dir_list[i]=='six':
       map[i][0]='six'
       map[i][1]=6
    elif data_dir_list[i]=='three':
       map[i][0]='three'
       map[i][1]=3
    elif data_dir_list[i]=='two':
       map[i][0]='two'
       map[i][1]=2
    elif data_dir_list[i]=='zero':
       map[i][0]='zero'
       map[i][1]=0
       
num_channel=1

num_classes = 10

audio_data_list=[]
size_data=[]
lengths=[]
zeros=np.zeros((13),dtype='int64')
zeros = zeros.astype('float32')

    
max=0
for dataset in data_dir_list:
    audio_list=os.listdir(data_path+'/'+ dataset)
    lengths.append(len(audio_list))
        

    for audio in audio_list:
        X, sample_rate = librosa.load(data_path+'/'+ dataset +'/'+audio, res_type='kaiser_fast')
        mfccs=np.array(librosa.feature.mfcc(y=X,sr=sample_rate, n_mfcc=13).T)
        mfccs=list(mfccs)
        while len(mfccs)<74:
              mfccs.append(zeros)
        audio_data_list.append(np.array(mfccs))
size_data.append(len(audio_list))
audio_data = np.array(audio_data_list)
audio_data = audio_data.astype('float32')
audio_data /= 255

if num_channel==1:
	if K.image_dim_ordering()=='th':
		audio_data=np.expand_dims(audio_data, axis=1)
		print(audio_data.shape)

	else: 
		audio_data=np.expand_dims(audio_data, axis=4)
		print(audio_data.shape)

num_classes = 10

num_of_samples = audio_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

i=1
while(i<10):
     lengths[i]=lengths[i]+lengths[i-1]
     i+=1

print(lengths)

for k in range(0,lengths[0]):
    labels[k]=map[0][1]


i=1
while(i<10):
     for j in range(lengths[i-1],lengths[i]):
         labels[j]=map[i][1]
     i+=1



Y= np_utils.to_categorical(labels,10)
x,y = shuffle(audio_data,Y, random_state=5)

X_train,X_val, y_train, y_val = train_test_split(x, y, test_size=0.30, random_state=4)
X_train,X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=4) 


shape=X_train[0].shape
print(shape)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=shape, padding='same', name='conv1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', name='conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3), padding='same', name='conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax', name='op'))

adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])

filepath = 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callback_list = [checkpoint]
model.fit(X_train,y_train, epochs=100, batch_size=32, callbacks=callback_list)

score = model.evaluate(X_test, y_test, verbose=0)
print('the testing accuracy is',score[1])
test_image = X_test

(model.predict(test_image))
print(model.predict_classes(test_image))

outputs = [layer.output for layer in model.layers]
print(outputs)

del model