from keras.models import model_from_json
from keras.layers import SeparableConv2D, Conv2D, Dense, BatchNormalization
from keras import regularizers
import os

def set_model_regularization(model, regularization_function, layers=None, bias_term=False):
    """
    Go through every Convolutional, Dense and BN layer in the model and apply the specified regularization function
    Args:
        model: Keras Machine Learning model
        regularization_function: Regularization function to include in layers (string)
        layers: Layers to which the regularization will be applied
        bias_term: If true will add the regularization in the layer bias regularizer

    Returns: Model with regularization function applied

    """
    # Save weights in case it is a pre-trained model to not lose them afterwards
    model.save_weights("tmp.h5")
    if regularization_function == 'l1':
        regularization_function = regularizers.l1(5e-4)
    elif regularization_function == 'l2':
        regularization_function = regularizers.l2(5e-4)
    else:
        raise ValueError("wrong input regularization_function")

    if layers is None:
        layers = model.layers
    # we only regularize the layers that are trainable. We ignore those layers that are not trainable
    for layer_index, layer in enumerate(layers):
        if isinstance(layer, SeparableConv2D) and layer.trainable:
            layer.depthwise_regularizer, layer.pointwise_regularizer = regularization_function, regularization_function
            if bias_term:
                layer.bias_regularizer = regularization_function
        elif (isinstance(layer, Conv2D) and layer.trainable) or (isinstance(layer, Dense) and layer.trainable):
            layer.kernel_regularizer = regularization_function
            if bias_term:
                layer.bias_regularizer = regularization_function
        elif isinstance(layer, BatchNormalization) and layer.trainable:
            layer.gamma_regularizer, layer.beta_regularizer = regularization_function, regularization_function

    # This workaround is needed to add the regularizations and recover the original weights
    model = model_from_json(model.to_json())
    model.load_weights("tmp.h5", by_name=True)
    os.system("rm tmp.h5")
    return model

