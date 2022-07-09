import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 100
img_width = 100

train_ds = tf.keras.utils.image_dataset_from_directory(
    "./dataset/entrenamiento",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "./dataset/validacion",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

resultList = {"imagen dada": [], "prediccion": []}
y = []
labels = []
x = 0

directoriesOfCategories = os.listdir('./dataset/entrenamiento/')

for category in directoriesOfCategories:
    for imagen in os.listdir('./dataset/entrenamiento/'+category):
        img = tf.keras.utils.load_img(
            './dataset/entrenamiento/'+category+"/"+imagen, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 
        predicciones = model.predict(img_array)
        y.append(np.argmax(predicciones[0]))
        prediccion = directoriesOfCategories[np.argmax(predicciones[0])]
        resultList["imagen dada"].append(imagen)
        resultList["prediccion"].append(prediccion)
        labels.append(x)
    x += 1


y = np.asanyarray(y)
yP = y[:]
df = pd.DataFrame(resultList)
df.to_csv('./output.csv')
matriz = tf.math.confusion_matrix(labels, yP)
figMatriz = plt.figure(figsize=(5, 5))
sns.heatmap(matriz, xticklabels=directoriesOfCategories,
            yticklabels=directoriesOfCategories, annot=True, fmt="g", cmap="mako")
plt.xlabel("prediccion")
plt.ylabel("entrada")
plt.show()