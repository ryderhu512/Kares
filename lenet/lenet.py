
# ----------------------------------------------------------------------------
# import kares modules
# ----------------------------------------------------------------------------
import keras
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import load_model
from keras.utils import np_utils
from keras.utils import plot_model

# ----------------------------------------------------------------------------
# import common and user-defined modules
# ----------------------------------------------------------------------------
import pickle
import numpy as np
import matplotlib.pyplot as plt   
import data_loader

# ----------------------------------------------------------------------------
# load data
# ----------------------------------------------------------------------------
n_training_data = 10000
n_test_data = 100

train_dataset,train_label = data_loader.load_training_data(n_training_data)
train_dataset = [np.array(m).reshape(28, 28, 1) for m in train_dataset]
train_dataset = np.array(train_dataset)
train_label = np.array(train_label)

test_dataset,test_label = data_loader.load_test_data(n_test_data)
test_dataset = [np.array(m).reshape(28, 28, 1) for m in test_dataset]
test_dataset = np.array(test_dataset)
test_label = np.array(test_label)

# ----------------------------------------------------------------------------
# build model
# ----------------------------------------------------------------------------
def build_lenet(train = 0):
    if train == 1:
        model = keras.models.Sequential()
        
        model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(28,28,1), \
                padding='valid',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(64,(5,5),strides=(1,1), \
                padding='valid',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  

        model.add(Flatten())  
        model.add(Dense(100,activation='relu'))  
        model.add(Dense(10,activation='softmax'))  

        model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
        hist=model.fit(x=train_dataset,y=train_label,batch_size=100,epochs=10,verbose=2,
                  validation_data=(test_dataset,test_label))
        model.save('lenet.h5')
        return model

    else:
        model = load_model('lenet.h5')
        return model

# ----------------------------------------------------------------------------
# test
# ----------------------------------------------------------------------------
model=build_lenet(train = 1)
result=model.predict(test_dataset, batch_size=32, verbose=1)
print(len([np.argmax(m) for m in result]))

loss_and_metrics=model.evaluate(test_dataset, test_label, batch_size=50)
print(loss_and_metrics)

plot_model(model, to_file='letnet.png')

'''
# analyze weights
weights = model.get_weights()
for w in weights:
    print(len(w))
    print(w[0])

m=np.array(weights[2])
print(len(m))
plt.imshow(m, cmap = 'binary')
plt.show()
'''
    

#print(hist.history)


# === END ===
