#helper function that splits the 3 images and shapes them into 28 x 28
def splitImages(X):
  top_numbers = X[:, : 784].reshape(-1, 28, 28, 1) / 255
  middle_numbers = X[:, 784: 1568].reshape(-1, 28, 28, 1) / 255
  bottom_numbers = X[:, 1568:].reshape(-1, 28, 28, 1) / 255
  return top_numbers, middle_numbers, bottom_numbers

def classify(Xtest, model):
    top_images, middle_images, bottom_images = splitImages(Xtest)

    #make the prediction
    preds = model.predict({'top_input': top_images, 'middle_input': middle_images, 'bottom_input': bottom_images})
    return preds.argmax(axis=1) #return the m x 1 vector of predictions