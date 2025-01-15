import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from helpers import splitImages, classify

def learn(X, y):
    #define the cnn
    def cnn(input_shape):
        model = models.Sequential([
            Input(input_shape),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(1, 1)),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(1, 1)),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
        ])
        return model

    image_shape = (28, 28, 1)

    #inputs for top, middle, and bottom images
    top_input = layers.Input(shape=image_shape, name="top_input")
    middle_input = layers.Input(shape=image_shape, name="middle_input")
    bottom_input = layers.Input(shape=image_shape, name="bottom_input")

    #cnn for all images
    cnn_model = cnn(image_shape)

    #extract the features from each image
    top_features = cnn_model(top_input)
    middle_features = cnn_model(middle_input)
    bottom_features = cnn_model(bottom_input)

    #predict if the top image is odd or even
    top_classification = layers.Dense(1, activation='sigmoid')(top_features)  #0 for even, 1 for odd

    #use the top image classification to choose between middle or bottom features
    selected_features = layers.Lambda(
        #x[0]: top classification (either 0 for even or 1 for odd)
        #x[1]: features for the middle number
        #x[2]: features for the bottom number

        #if top_classification is even, the first term becomes 0 and x[2] is used
        #if top_classification is odd, second term becomes 0 and x[1] is used
        lambda x: x[0] * x[1] + (1 - x[0]) * x[2]
    )([top_classification, middle_features, bottom_features])

    #final dense layers for classification
    output = layers.Dense(10, activation='softmax', name="output")(selected_features)

    model = models.Model(inputs=[top_input, middle_input, bottom_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #prepare data and fit model
    top_images, middle_images, bottom_images = splitImages(X)
    model.fit({'top_input': top_images, 'middle_input': middle_images, 'bottom_input': bottom_images}, y, epochs=3, batch_size=64)

    return model

def main():
    data = pd.read_csv(r'train.csv', header=None).to_numpy()
    validation = pd.read_csv(r'validation.csv', header=None).to_numpy()

    Y_train = data[:, 0]
    X_train = data[:, 1:]

    Y_val = validation[:, 0]
    X_val = validation[:, 1:]

    model = learn(X_train, Y_train) #get the model

    yhat = classify(X_val, model) #get predictions on validation set

    print(f"Accuracy: {100 * np.mean(yhat == Y_val):1f}%")

if __name__ == "__main__":
    main()
