import model_mobilenet as mn
import tensorflow as tf
#from keras import backend as K
from keras.optimizers import Nadam, Adam, RMSprop, SGD

def get_model(argument, classes, net_param):
    return mn.MobileNet(input_shape=(224, 224, 3), weights=None, classes=classes, alpha=net_param, mode=argument)

def get_optimizer(configuration, lr=None):
    switcher = {
        1: Nadam(lr=0.001 if lr is None else lr),
        2: Nadam(lr=0.0001 if lr is None else lr),
        3: Nadam(lr=0.001 if lr is None else lr),
        4: Nadam(lr=0.0001 if lr is None else lr),
	5: RMSprop(lr=0.045 if lr is None else lr, rho=0.9, decay=0.9),
        6: tf.train.RMSPropOptimizer(0.045 if lr is None else lr, 0.9, 0.9),
        7: SGD(lr=0.045, momentum=0.9, decay=0.00004, nesterov=True)
    }
    optimizer = switcher.get(configuration, None)
#    print(optimizer.lr)
#    print(K.get_value(optimizer.lr))
    return optimizer
