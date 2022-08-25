# Loading required libraries and functions
import os
import shutil
import random
import itertools
import base64
from io import BytesIO
# %matplotlib inline
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from keras import backend
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Activation
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
import flask
from flask import Flask, jsonify, request 
from flask_cors import CORS
from io import BytesIO
# from IPython.display import Image

# Loading data and preprocessing images according to mobilenet requirements
# Creating batches of data

labels = ['Flooding', 'No Flooding']
# train_path = 'data/train'
# valid_path = 'data/valid'
# test_path = 'data/test'

# train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
#     directory=train_path, target_size=(224,224), batch_size=10)
# valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
#     directory=valid_path, target_size=(224,224), batch_size=10)
# test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
#     directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

# #Loading pre-trained lightweight mobilenet image classifier
# mobile = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False)
# # mobile.summary()

# # Store all layers of the original mobilenet except the last 5 layers in variable x
# # There is no predefined logic behind this, it just gives the optimal results for this task
# # Also, we will be only training the last 12 layers of the mobilenet during finetuning as we want 
# # it to keep all of the previously learned weights 
# x = mobile.layers[-12].output
# print(x)

# # Create global pooling, dropout and a binary output layer, as we want our model to be a binary classifier, 
# # i.e. to classify flooding and no flooding
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
# output = Dense(units=2, activation='sigmoid')(x)

# # Construct the new fine-tuned mode
# model = Model(inputs=mobile.input, outputs=output)

# # Freez weights of all the layers except for the last five layers in our new model, 
# # meaning that only the last 12 layers of the model will be trained.
# for layer in model.layers[:-23]:
#     layer.trainable = False

# model.summary()

# # Compile the model
# model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(x=train_batches,
#           steps_per_epoch=len(train_batches),
#           validation_data=valid_batches,
#           validation_steps=len(valid_batches),
#           epochs=10,
#           verbose=2
# )

# # Saving and loading our trained for future use

# model.save("fine_tuned_flood_detection_model")
# # model.load_weights('fine_tuned_flood_detection_model')

# # Make predictions and plot confusion matrix to look how well our model performed in classifying 
# # flooding and no flooding images 

# test_labels = test_batches.classes
# predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
# cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
# precision = precision_score(y_true=test_labels, y_pred=predictions.argmax(axis=1))
# f1_score = f1_score(y_true=test_labels, y_pred=predictions.argmax(axis=1))
# accuracy = accuracy_score(y_true=test_labels, y_pred=predictions.argmax(axis=1))
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#             horizontalalignment="center",
#             color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# # Pring precision, F1 score and accuracy of our model
# print('Precision: ', precision)
# print('F1 Score: ', f1_score)
# print('Accuracy: ', accuracy)

# # Confusion Matrix 
# test_batches.class_indices
# cm_plot_labels = ['Flooding','No Flooding']
# plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# # Prepare image for mobilenet prediction

def preprocess_image(file):
    img_path = 'upload/'
    starter = file.find(',')
    file = file[starter+1:]
    with open("upload/img.png", "wb") as fh:
        fh.write(base64.decodebytes(file.encode('ascii')+ b'=='))
    img = image.load_img(img_path+"/img.png", target_size=(224, 224))
    # print(file)
    # img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)




app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    if 'imgBase64' not in request.get_json():
        return "Please try again. The Image doesn't exist"
    
    # Display image which we want to predict
    # Image(filename='evaluate/3.jpeg', width=300,height=200) 

    # Preprocess image and make prediction
    payload = request.get_json()
    # print (payload)
    preprocessed_image = preprocess_image(payload['imgBase64'])
    model = tf.keras.models.load_model('fine_tuned_flood_detection_model')
    predictions = model.predict(preprocessed_image)

    # Print predicted accuracy scores for both classes, i.e. (1) Flooding, (2) No Flooding
    print(predictions)

    # Get the maximum probability score for predicted class from predictions array
    result = np.argmax(predictions)

    # Print the predicted class label
    print(labels[result])

    # train_batches[0][1][1]

    # Return on a JSON format
    return jsonify(prediction=labels[result])
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')