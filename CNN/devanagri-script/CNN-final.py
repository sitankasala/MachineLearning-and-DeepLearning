import numpy as np
import pandas as pd
import os as os

DATA_DIR='D:\\data science\\Repository\\CNN\\devanagri-script\\Data\\'
os.chdir(DATA_DIR)

train_df = pd.read_csv('Train.csv')
train_labels_str = train_df['Label']
train_df.drop('Label',axis=1,inplace=True)
train_data_arr = train_df.values/255.0
train_data_arr = train_data_arr.reshape(train_df.shape[0],32,32,1)


from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D,BatchNormalization
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
encoder = LabelEncoder()
y=encoder.fit_transform(train_labels_str)
target=to_categorical(y,train_labels_str.nunique())
all_labels = train_labels_str.unique()

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(train_data_arr, target, train_size=0.80,test_size=0.20, random_state=400,stratify=target)

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=X_train[0].shape,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
model.add(Flatten())
model.add(Dense(128,activation='relu',kernel_initializer='uniform'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu',kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Dense(46,activation='softmax',kernel_initializer='uniform'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy']) 
model.summary()

history = model.fit(X_train, y_train,
  batch_size=256, epochs=30,
  validation_data=(X_test, y_test))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'],loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'],loc='upper left')
plt.show()

y_preds = np.argmax(model.predict_proba(X_test), axis=1)
y_tests = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
train_labels = encoder.fit_transform(all_labels)
conf_mat = confusion_matrix(y_tests, y_preds,train_labels)
fig,ax= plot_confusion_matrix(conf_mat=conf_mat,figsize=(20,20),show_absolute=True)
plt.show()
class_report = classification_report(y_tests, y_preds, target_names=all_labels)


test_data = pd.read_csv("D:\\data science\\Repository\\CNN\\devanagri-script\\Data\\test_X.csv")
test_data.shape
test_data_arr = test_data.values/255.0
test_data_arr = test_data_arr.reshape(test_data.shape[0],32,32,1)
test_data_arr.shape
y_pred = model.predict_proba(test_data_arr)
predictions = []
for row in y_pred:
    predictions.append(all_labels[np.argmax(row)])

pred_df = pd.DataFrame(predictions, columns=['Label']).to_csv('../pred.csv',index=False)


