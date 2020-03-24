from keras_vggface.vggface import VGGFace
from keras_contrib.applications import ResNet # keras-contrib is required, to install: https://github.com/keras-team/keras-contrib#install-keras_contrib-for-keras-teamkeras
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model, load_model
import os
"""
only the resnet50 model has the imagenet weights;
resnet18 and resnet34 model do not have weights
"""

class ModelFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_by_name(model_name, opt, num_classes):
        input_shape = (opt.image_size, opt.image_size, 3)
        weights_source = None
        if len(opt.checkpoint_path)==0:
            if model_name == 'ResNet50':
                base_model = VGGFace(model='resnet50') 
                weights_source = 'VGGFace2'
            elif model_name == 'SeNet50':
                base_model = VGGFace(model='senet50')
                weights_source = 'VGGFace2'
            elif model_name == 'ResNet18':
                """ResNet with 18 layers and v2 residual units
                """                
                base_model = ResNet(input_shape, num_classes, 'basic', repetitions=[2, 2, 2, 2])
                weights_source = 'scratch'
            elif model_name == 'ResNet34':
                """ResNet with 34 layers and v2 residual units
                """
                base_model = ResNet(input_shape, num_classes, 'basic', repetitions=[3, 4, 6, 3])
                weights_source = 'scratch'
            base_model.layers.pop() # remove the final dense layer
            x = base_model.layers[-1].output
            if opt.dropout_rate > 0.:
                x = Dropout(opt.dropout_rate, name='dropout_layer')(x)
            x = Dense(num_classes, name='dense', activation = opt.activation)(x)
            model = Model(base_model.input, x)
            print("model {} was built from {}.".format(model_name, weights_source))
        else:
            model_path = opt.checkpoint_path
            assert os.path.exists(model_path), "model {} does not exist.".format(model_path)
            model = load_model(model_path)
            print("model {} was loaded from {}.".format(model_name, model_path))
        if (opt.freeze_before !=0) and opt.is_train:
            pos_id = opt.freeze_before > 0 
            for id, layer in enumerate(model.layers):
                if (pos_id and (id + 1) > opt.freeze_before) or \
                ((not pos_id) and id-len(model.layers)<opt.freeze_before):
                    layer.trainable=False
                else:
                    pass

        model.summary()
        if (opt.freeze_before !=0) and opt.is_train:
            get_layer = model.layers[opt.freeze_before]
            print("Freeze parameter before the layer: {}".format(get_layer.name))
        return model
