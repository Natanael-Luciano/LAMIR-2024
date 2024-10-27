
#!pip install kapre

import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import LSTM
from keras.layers import Reshape
from keras.layers import Permute
from keras.layers import Bidirectional

from keras.optimizers import Adam

from tensorflow.keras.metrics import AUC

from tensorflow.keras.backend import l2_normalize

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import kapre
from kapre.composed import get_melspectrogram_layer
from keras.callbacks import CSVLogger



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape, Conv2D, add

import tensorflow as tf

def model_CNN_HK(sr,input_shape,n_fft=512,n_mels=128,hop_length=256,n_feat=32, L2 = False):
  """
  Modelo rede convolucional com e sem normalização L2 e kernel horizontal
  """
  k=1
  while n_fft + k*hop_length<sr:
    k+=1
  #print(k)
  model = Sequential()
  # A mel-spectrogram layer

  model.add(get_melspectrogram_layer(input_shape=input_shape,n_fft=n_fft,
                          hop_length=hop_length,
                          sample_rate=sr, n_mels=n_mels,
                          mel_f_min=0.0, mel_f_max=sr/2,
                          return_decibel=True,
                          name='nt_melspec'))

  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(k, 3),
                  strides = (k//4,1),
                  padding='valid',
                  name='conv1',
                  activation='relu'))
  model.add(BatchNormalization(name='bnorm1'))
  model.add(Activation('relu'))

  model.add(MaxPool2D(pool_size=(2, 2),
                      strides = (2,2),
                      padding='valid',
                      name='maxpool1'))

  # model.add(BatchNormalization(name='bnorm1'))

  model.add(Conv2D(filters=16,
                  # kernel_size=(1, 4),
                  kernel_size=(10, 3),
                  padding='valid',
                  name='conv2',
                  activation='relu'))
  model.add(BatchNormalization(name='bnorm2'))
  model.add(Activation('relu'))

  model.add(MaxPool2D(pool_size=(2, 2),
                      strides = (2,2),
                      padding='valid',
                      name='maxpool2'))

  model.add(Conv2D(filters=8,
                  # kernel_size=(128, 4),
                  kernel_size=(5, 3),
                  padding='valid',
                  name='conv3',
                  activation='relu'))

  model.add(MaxPool2D(pool_size=(2, 2),
                      strides = (2,2),
                      padding='valid',
                      name='maxpool3'))

  model.add(BatchNormalization(name='bnorm3'))
  model.add(Activation('relu'))

  model.add(BatchNormalization(name='bnorm4'))
  model.add(Activation('relu'))

  model.add(Flatten(name='flat'))

  model.add(BatchNormalization(name='bnorm5'))

  model.add(Dense(units=32,
                  activation='sigmoid',
                  name='dense2'))
  if L2:
    model.add(Lambda(lambda x: l2_normalize(x - 0.5, axis=1) + 0.5))

  model.add(Dropout(0.1))

  model.add(Dense(units=16,
                  activation='sigmoid',
                  name='dense3'))
  if L2:
    model.add(Lambda(lambda x: l2_normalize(x - 0.5, axis=1) + 0.5))

  model.add(Dropout(0.1))

  model.add(Dense(units=10,
                  activation='softmax',
                  name='output'))

  model.layers[0].trainable = False
  #model.layers[1].trainable = False

  model.build(input_shape=(None, ))
  model.summary()

  return model

if __name__=="__main__":
  model1 = model_CNN_HK(sr=22050,input_shape=(660000,1),n_fft=512,n_mels=512,hop_length=256,n_feat=64)

def model_CNN_VK(sr,input_shape,n_fft=512,n_mels=128,hop_length=256,n_feat=32, L2 = False):
  """
  Modelo rede convolucional com e sem normalização L2 e kernel vertical
  """
  model = Sequential()
  # A mel-spectrogram layer
  model.add(get_melspectrogram_layer(n_fft=n_fft, hop_length=hop_length, input_shape=input_shape,
                          sample_rate=sr, n_mels=n_mels,
                          mel_f_min=0.0, mel_f_max=sr/2,
                          return_decibel=True,
                          name='nt_melspec'))

  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(3,n_mels//2),
                  strides = (1,10),
                  padding='valid',
                  name='conv1',
                  activation='relu'))
  model.add(BatchNormalization(name='bnorm0'))
  model.add(Activation('relu'))

  model.add(MaxPool2D(pool_size=(2, 2),
                    strides = (2,2),
                     padding='valid',
                     name='maxpool1'))

  model.add(BatchNormalization(name='bnorm1'))
  model.add(Activation('relu'))

  model.add(Conv2D(filters=16,
                  # kernel_size=(3, 3),
                  kernel_size=(1, 1),
                  padding='valid',
                  name='conv2',
                  activation='relu'))
  model.add(BatchNormalization(name='bnorm2'))
  model.add(Activation('relu'))

  model.add(MaxPool2D(pool_size=(20, 1),
                      strides = (1,1),
                      padding='valid',
                      name='maxpool2'))

  model.add(Conv2D(filters=8,
                  # kernel_size=(128, 4),
                  kernel_size=(10, 1),
                  padding='valid',
                  name='conv3',
                  activation='relu'))

  model.add(MaxPool2D(pool_size=(12, 1),
                      strides = (10,1),
                      padding='valid',
                      name='maxpool3'))


  model.add(BatchNormalization(name='bnorm3'))
  model.add(Activation('relu'))

  model.add(BatchNormalization(name='bnorm4'))
  model.add(Activation('relu'))

  model.add(Flatten(name='flat'))

  model.add(BatchNormalization(name='bnorm5'))

  model.add(Dense(units=32,
                  activation='sigmoid',
                  name='dense2'))
  if L2:
    model.add(Lambda(lambda x: l2_normalize(x - 0.5, axis=1) + 0.5))

  model.add(Dropout(0.1))

  model.add(Dense(units=16,
                  activation='sigmoid',
                  name='dense3'))
  if L2:
    model.add(Lambda(lambda x: l2_normalize(x - 0.5, axis=1) + 0.5))

  model.add(Dropout(0.1))

  model.add(Dense(units=10,
                  activation='softmax',
                  name='output'))

  model.layers[0].trainable = False

  model.summary()

  return model

if __name__=="__main__":
  model1 = model_CNN_VK(sr=22050,input_shape=(660000,1),n_fft=512,n_mels=512,hop_length=256,n_feat=64)

def model_CNN_SK(sr,input_shape,n_fft=512,n_mels=128,hop_length=256,n_feat=32, L2 = False):
  """
  Modelo rede convolucional com e sem normalização L2 e kernel quadrado
  """
  model = Sequential()
  
  # A mel-spectrogram layer
  model.add(get_melspectrogram_layer(n_fft=n_fft, hop_length=hop_length, input_shape=input_shape,
                          sample_rate=sr, n_mels=n_mels,
                          mel_f_min=0.0, mel_f_max=sr/2,
                          return_decibel=True,
                          name='nt_melspec'))

  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(10,10),
                  strides = (1,1),
                  padding='valid',
                  name='conv1',
                  activation='relu'))
  model.add(BatchNormalization(name='bnorm1'))
  model.add(Activation('relu'))

  model.add(MaxPool2D(pool_size=(2, 2),
                      strides = (2,2),
                      padding='valid',
                      name='maxpool1'))

  model.add(Conv2D(filters=8,
                  # kernel_size=(3, 3),
                  kernel_size=(10,10 ),
                  padding='valid',
                  name='conv2',
                  activation='relu'))

  model.add(BatchNormalization(name='bnorm2'))
  model.add(Activation('relu'))

  model.add(MaxPool2D(pool_size=(2, 2),
                      strides = (2,2),
                      padding='valid',
                      name='maxpool2'))

  model.add(Conv2D(filters=4,
                  # kernel_size=(128, 4),
                  kernel_size=(10, 10),
                  padding='valid',
                  name='conv3',
                  activation='relu'))

  model.add(MaxPool2D(pool_size=(2, 2),
                      strides = (2,2),
                      padding='valid',
                      name='maxpool3'))

  model.add(BatchNormalization(name='bnorm3'))
  model.add(Activation('relu'))
  model.add(Conv2D(filters=4,
                  # kernel_size=(128, 4),
                  kernel_size=(10, 10),
                  padding='valid',
                  name='conv4',
                  activation='relu'))

  model.add(MaxPool2D(pool_size=(2, 2),
                      strides = (2,2),
                      padding='valid',
                      name='maxpool4'))


  model.add(BatchNormalization(name='bnorm4'))
  model.add(Activation('relu'))

  model.add(Flatten(name='flat'))

  model.add(Dense(units=32,
                  activation='relu',
                  name='dense1'))
  if L2:
    model.add(Lambda(lambda x: l2_normalize(x - 0.5, axis=1) + 0.5))

  model.add(Dropout(0.1))

  model.add(Dense(units=16,
                  activation='relu',
                  name='dense2'))
  if L2:
    model.add(Lambda(lambda x: l2_normalize(x - 0.5, axis=1) + 0.5))

  model.add(Dropout(0.1))

  model.add(Dense(units=10,
                  activation='softmax',
                  name='output'))

  model.layers[0].trainable = False

  model.summary()

  return model

if __name__=="__main__":
  model1 = model_CNN_SK(sr=22050,input_shape=(660000,1),n_fft=512,n_mels=512,hop_length=256,n_feat=32)

def model_CRNN_VK(sr,input_shape,n_fft=512,n_mels=512,hop_length=256,n_feat=64):
  """
  Modelo rede convolucional recorrente e kernel vertical
  """

  model = Sequential()

  model.add(Input(shape=input_shape, dtype=tf.float32))  # Usar Input layer

  # A mel-spectrogram layer
  model.add(get_melspectrogram_layer(n_fft=n_fft, hop_length=hop_length, input_shape=input_shape,
                          sample_rate=sr, n_mels=n_mels,
                          mel_f_min=0.0, mel_f_max=sr/2,
                          return_decibel=True,
                          name='nt_melspec',
                          input_data_format='channels_last',
                          output_data_format='default',))
  # nmels= #número de linhas no spectrograma (aumenta eleeee)
  # output é da forma  (batch, time  , FREQ , Chanels) por isso o kernel da conv
  #  ta deitado
  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(8, n_mels//2),
                  strides = (1,n_mels//4),
                  padding='same',
                  name='conv1'))# ,
                  # activation='relu'))

  model.add(BatchNormalization(name='bnorm0'))
  model.add(Activation('relu'))


  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(8, 4),
                  strides = (1,1),
                  padding='valid',
                  name='conv2'))# ,
                  # activation='relu'))
  model.add(BatchNormalization(name='bnorm1'))
  model.add(Activation('relu'))


  #add reshape, cada filtro retorna um vetor da forma ("tempo",1)
  #precisamos modificar para ("tempo",filtros)

  model.add(Reshape((-1,n_feat,1)))

  #model.add(Permute((2,1)))
  #model.add(Lambda(lambda x: tf.expand_dims(x, -1).shape))

  model.add(Reshape((-1,n_feat)))

  model.add(LSTM(64,
          return_sequences=False))

  model.add(Dense(units=10,
                  activation='softmax',
                  name='output'))

  model.layers[0].trainable = False
  #model.layers[4].trainable = False
  #model.layers[5].trainable = False

  model.summary()

  return model

if __name__=="__main__":
  model1 = model_CRNN_VK(sr=22050,input_shape=(220000,1),n_fft=512,n_mels=512,hop_length=256,n_feat=64)

def model_BICRNN_VK(sr,input_shape,n_fft=512,n_mels=512,hop_length=256,n_feat=64):
  """
  Modelo rede convolucional recorrente e kernel vertical
  """
  model = Sequential()
  # A mel-spectrogram layer
  model.add(get_melspectrogram_layer(n_fft=n_fft, hop_length=hop_length, input_shape=input_shape,
                          sample_rate=sr, n_mels=n_mels,
                          mel_f_min=0.0, mel_f_max=sr/2,
                          return_decibel=True,
                          name='nt_melspec',
                          input_data_format='channels_last',
                          output_data_format='default',))
  # nmels= #número de linhas no spectrograma (aumenta eleeee)
  # output é da forma  (batch, time  , FREQ , Chanels) por isso o kernel da conv
  #  ta deitado
  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(8, n_mels//2),
                  strides = (1,n_mels//4),
                  padding='same',
                  name='conv1'))# ,
                  # activation='relu'))

  model.add(BatchNormalization(name='bnorm0'))
  model.add(Activation('relu'))


  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(8, 4),
                  strides = (1,1),
                  padding='valid',
                  name='conv2'))# ,
                  # activation='relu'))
  model.add(BatchNormalization(name='bnorm1'))
  model.add(Activation('relu'))


  #add reshape, cada filtro retorna um vetor da forma ("tempo",1)
  #precisamos modificar para ("tempo",filtros)

  model.add(Reshape((-1,n_feat,1)))

  #model.add(Permute((2,1)))
  #model.add(Lambda(lambda x: tf.expand_dims(x, -1).shape))

  model.add(Reshape((-1,n_feat)))

  model.add(Bidirectional(LSTM(64,
          return_sequences=False)))

  model.add(Dense(units=10,
                  activation='softmax',
                  name='output'))

  model.layers[0].trainable = False
  #model.layers[4].trainable = False
  #model.layers[5].trainable = False

  model.summary()

  return model

if __name__=="__main__":
  model1 = model_BICRNN_VK(sr=22050,input_shape=(660000,1),n_fft=512,n_mels=512,hop_length=256,n_feat=64)

def model_CRNN_VEC_VK(sr,input_shape,n_fft=512,n_mels=512,hop_length=256,n_feat=64):
  """
  Modelo rede convolucional recorrente e kernel vertical
  """
  model = Sequential()
  # A mel-spectrogram layer
  model.add(get_melspectrogram_layer(n_fft=n_fft, hop_length=hop_length, input_shape=input_shape,
                          sample_rate=sr, n_mels=n_mels,
                          mel_f_min=0.0, mel_f_max=sr/2,
                          return_decibel=True,
                          name='nt_melspec',
                          input_data_format='channels_last',
                          output_data_format='default',))
  # nmels= #número de linhas no spectrograma (aumenta eleeee)
  # output é da forma  (batch, time  , FREQ , Chanels) por isso o kernel da conv
  #  ta deitado
  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(8, n_mels//2),
                  strides = (1,n_mels//4),
                  padding='same',
                  name='conv1'))# ,
                  # activation='relu'))

  model.add(BatchNormalization(name='bnorm0'))
  model.add(Activation('relu'))


  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(8, 4),
                  strides = (1,1),
                  padding='valid',
                  name='conv2'))# ,
                  # activation='relu'))
  model.add(BatchNormalization(name='bnorm1'))
  model.add(Activation('relu'))



  #add reshape, cada filtro retorna um vetor da forma ("tempo",1)
  #precisamos modificar para ("tempo",filtros)

  model.add(Reshape((-1,n_feat,1)))

  model.add(Conv2D(filters=1,
                  # kernel_size=(128, 4),
                  kernel_size=(8, n_feat),
                  strides = (1,1),
                  padding='valid',
                  name='conv3'))# ,
                  # activation='relu'))

  #model.add(Permute((2,1)))
  #model.add(Lambda(lambda x: tf.expand_dims(x, -1).shape))

  model.add(Reshape((-1,1)))



  model.add(LSTM(64,
          return_sequences=False))

  model.add(Dense(units=10,
                  activation='softmax',
                  name='output'))

  model.layers[0].trainable = False
  #model.layers[4].trainable = False
  #model.layers[5].trainable = False

  model.summary()

  return model

if __name__=="__main__":
  sr=22050
  input_shape=(220000,1)
  n_fft=2048
  n_mels= 256
  hop_length=1024
  n_feat=128

  model1 = model_CRNN_VEC_VK(sr,input_shape,n_fft,n_mels,hop_length,n_feat)









if __name__=="__main__":
  model1 = model_CRNN_VK(sr=22050,input_shape=(660000,1),n_fft=512,n_mels=512,hop_length=256,n_feat=64)



def model_BICRNN_VK_concat(sr,input_shape,n_fft=512,n_mels=512,hop_length=256,n_feat=64):
  """
  Modelo rede convolucional recorrente e kernel vertical
  """
  model = Sequential()
  # A mel-spectrogram layer
  model.add(get_melspectrogram_layer(n_fft=n_fft, hop_length=hop_length, input_shape=input_shape,
                          sample_rate=sr, n_mels=n_mels,
                          mel_f_min=0.0, mel_f_max=sr/2,
                          return_decibel=True,
                          name='nt_melspec',
                          input_data_format='channels_last',
                          output_data_format='default',))
  # nmels= #número de linhas no spectrograma (aumenta eleeee)
  # output é da forma  (batch, time  , FREQ , Chanels) por isso o kernel da conv
  #  ta deitado
  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(8, n_mels//2),
                  strides = (1,n_mels//4),
                  padding='same',
                  name='conv1'))# ,
                  # activation='relu'))

  model.add(BatchNormalization(name='bnorm0'))
  model.add(Activation('relu'))


  model.add(Conv2D(filters=n_feat,
                  # kernel_size=(128, 4),
                  kernel_size=(8, 4),
                  strides = (1,1),
                  padding='valid',
                  name='conv2'))# ,
                  # activation='relu'))
  model.add(BatchNormalization(name='bnorm1'))
  model.add(Activation('relu'))


  #add reshape, cada filtro retorna um vetor da forma ("tempo",1)
  #precisamos modificar para ("tempo",filtros)

  model.add(Reshape((-1,n_feat,1)))

  #model.add(Permute((2,1)))
  #model.add(Lambda(lambda x: tf.expand_dims(x, -1).shape))

  model.add(Reshape((-1,n_feat)))

  model.add(Bidirectional(LSTM(64,
          return_sequences=False)))

  model.add(Dense(units=10,
                  activation='softmax',
                  name='output'))

  model.layers[0].trainable = False
  #model.layers[4].trainable = False
  #model.layers[5].trainable = False

  model.summary()

  return model



def parallel_convolution_lstm_model(sr,input_shape,n_fft=512,n_mels=512,hop_length=256,n_feat=32, lstm_units = 32, num_classes=10):
    # Input layer
    input_data = Input(shape=input_shape)

    # Kapre Melspectrogram layer (non-trainable)
    mel_spectrogram = get_melspectrogram_layer(input_shape=input_shape,
                          n_fft=n_fft, hop_length=hop_length,
                          sample_rate=sr, n_mels=n_mels,
                          mel_f_min=0.0, mel_f_max=sr/2,
                          return_decibel=True,
                          name='nt_melspec',
                          input_data_format='channels_last',
                          output_data_format='default')(input_data)

    mel_spectrogram.trainable = False

    # Reshape for parallel convolutions
    #reshaped_input = Reshape((mel_spectrogram.shape[1], mel_spectrogram.shape[2], 1))(mel_spectrogram)

    # First parallel convolutional layer
    conv1a = Conv2D(n_feat, kernel_size=(8, n_mels//2),
                  strides = (1,n_mels//4),
                  padding='valid',
                  name='conv1a')(mel_spectrogram)


    # Second parallel convolutional layer
    conv1b = Conv2D(n_feat, kernel_size=(8, n_mels),
                  strides = (1,1),
                  padding='valid',
                  name='conv1b')(mel_spectrogram)

    # Concatenate the outputs of the two parallel layers
    summed_output = add([conv1a, conv1b])

    # Reshape for LSTM input (assuming time dimension is the first axis)
    reshaped_sum = Reshape((summed_output.shape[2],summed_output.shape[1] ,summed_output.shape[3]))(summed_output)
    reshaped_sum2 = Reshape((-1,reshaped_sum.shape[2]))(reshaped_sum)


    #reshaped = Reshape((-1, summed_output.shape[-2]))(reshaped_sum)
    # LSTM layer with three features for each time step
    lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=False, ))(reshaped_sum2)

    # Final Dense layer with 10 neurons and softmax activation
    output_layer = Dense(num_classes, activation='softmax')(lstm_layer)

    # Create the final model
    model = Model(inputs=input_data, outputs=output_layer, name='parallel_convolution_lstm_model')
    model.summary()
    return model

if __name__=="__main__":
  model =parallel_convolution_lstm_model(sr = 22050,input_shape=(220000,1),n_fft=512,n_mels=512,hop_length=256,n_feat=32, lstm_units = 32, num_classes=10)
