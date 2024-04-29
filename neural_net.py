from matplotlib import pyplot as plt
import training

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = training.forward_prop(W1, b1, W2, b2, X)
    predictions = training.get_predictions(A2)
    return predictions

def test_prediction(X_train, Y_train, index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.text(-4.5, -1.5, "Prediction: "+str(prediction)+"\n"+"Label: "+str(label), bbox=dict(facecolor='red', alpha=0.5), fontsize=12)
    plt.show()