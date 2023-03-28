
import numpy as np

cost_sensitive = False
class_weights = np.array([50, 1]) # modify according to sample sizes
activation_function = "relu"
optimization = "adam"

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Activation, Dense, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import get_custom_objects
import time
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, cohen_kappa_score, mean_squared_error
from sklearn.metrics import precision_recall_curve, auc
from sklearn.svm import *
import hyperopt
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
import pandas as pd

print("\nImported Package Versions:\n\n-------------------------------\n")
print(f"Tensorflow: {tf.__version__}")
print(f"Numpy: {np.__version__}")
print(f"Scikit Learn: {sklearn.__version__}")
print(f"Hyperopt: {hyperopt.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"\n")

# define additional classes and funcitons -----------------------------------------------------------------------------------

def swish(x, beta = 1):
    return(x * K.sigmoid(beta*x))

get_custom_objects().update({"swish": Activation(swish)})

def mse_calc(test, pred_probs):
    output = np.zeros((pred_probs.shape[0], 1), dtype = float)
    for i in range(pred_probs.shape[0]):
        probs = pred_probs[i]
        target = test[i]
        if target == 0:
            output[i] == probs[0]
        else:
            output[i] = probs[1]
    mse = mean_squared_error(test, output)
    return(mse)

# define cyclic learning rate class

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
   
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

#

# declare VNIR CNSVM class -----------------------------------------------------------------------------------

class CNSVM_VNIR:
    
    def __init__(self, seed = 666, cost_sensitive = False):
        self.seed = seed
        self.cost_sensitive = cost_sensitive
        np.random.seed(self.seed)
    
    def prepare_training_data(self, file_name, sep, test_size = 0.3):
        
        self.dataset = np.loadtxt(file_name, delimiter = sep)
        self.X = self.dataset[:,0:94]
        self.Y = self.dataset[:,94]
        
        if self.cost_sensitive:
            pos_cases = self.Y[self.Y == 0].shape[0]
            neg_cases = self.Y[self.Y == 1].shape[0]
            self.bias_init = np.log(pos_cases / neg_cases)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y,
                                                                                test_size = test_size,
                                                                                random_state = self.seed)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
    
    def find_GPU_name(self):
        gpu_name = tf.config.experimental.list_logical_devices("GPU")
        return(gpu_name[0])
    
    def define_model(self, select_act = "relu", opt = "adam"):
        
        if select_act == "relu":
            self.act = "relu"
        elif select_act == "swish":
            self.act = "swish"
        else:
            return(print("Select a valid activation function between ReLU and Swish"))
        
        visible = Input(shape = (94,1))
        
        # Inception Module 1
        
        conv1 = Conv1D(64,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (visible)
        batch1 = BatchNormalization() (conv1)
        act1 = Activation(self.act) (batch1)
        
        conv2 = Conv1D(96,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (visible)
        batch2 = BatchNormalization() (conv2)
        act2 = Activation(self.act) (batch2)
        conv2_1 = Conv1D(128,
                        kernel_size = 3,
                        padding = "same",
                        kernel_initializer = "lecun_normal",
                        kernel_regularizer = l2(l = 0.0001)) (act2)
        batch2_1 = BatchNormalization() (conv2_1)
        act2_1 = Activation(self.act) (batch2_1)
        
        conv3 = Conv1D(16,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (visible)
        batch3 = BatchNormalization() (conv3)
        act3 = Activation(self.act) (batch3)
        conv3_1 = Conv1D(32,
                    kernel_size = 5,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (act3)
        batch3_1 = BatchNormalization() (conv3_1)
        act3_1 = Activation(self.act) (batch3_1)
        
        pool = MaxPooling1D(pool_size = 3, strides = 1, padding = "same") (visible)
        conv4 = Conv1D(32,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (pool) 
        batch4 = BatchNormalization() (conv4)
        act4 = Activation(self.act) (batch4)
        
        out = concatenate([act1, act2_1, act3_1, act4], axis = -1)
        
        # Inception Module 2
        
        conv5 = Conv1D(64,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (out)
        batch5 = BatchNormalization() (conv5)
        act5 = Activation(self.act) (batch5)
        
        conv6 = Conv1D(96,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (out)
        batch6 = BatchNormalization() (conv6)
        act6 = Activation(self.act) (batch6)
        conv6_1 = Conv1D(128,
                        kernel_size = 3,
                        padding = "same",
                        kernel_initializer = "lecun_normal",
                        kernel_regularizer = l2(l = 0.0001)) (act6)
        batch6_1 = BatchNormalization() (conv6_1)
        act6_1 = Activation(self.act) (batch6_1)
        
        conv7 = Conv1D(16,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (out)
        batch7 = BatchNormalization() (conv7)
        act7 = Activation(self.act) (batch7)
        conv7_1 = Conv1D(32,
                    kernel_size = 5,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (act7)
        batch7_1 = BatchNormalization() (conv7_1)
        act7_1 = Activation(self.act) (batch7_1)
        
        pool_2 = MaxPooling1D(pool_size = 3, strides = 1, padding = "same") (out)
        conv8 = Conv1D(32,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (pool_2)
        batch8 = BatchNormalization() (conv8)
        act8 = Activation(self.act) (batch8)

        out2= concatenate([act5, act6_1, act7_1, act8], axis = -1)
        
        flat = Flatten() (out2)
        drop1 = Dropout(0.54) (flat)
        hidden1 = Dense(200, activation = self.act) (drop1)
        drop2 = Dropout(0.33) (hidden1)
        hidden2 = Dense(150, activation = self.act) (drop2)
        drop3 = Dropout(0.1) (hidden2)
        hidden3 = Dense(100, activation = self.act) (drop3)
        drop4 = Dropout(0.46) (hidden3)
        hidden4 = Dense(50, activation = self.act) (drop4)
        
        if self.cost_sensitive:
            output = Dense(1, activation = "sigmoid", bias_initializer = self.bias_init) (hidden4)
        else:
            output = Dense(1, activation = "sigmoid") (hidden4)
        
        self.model = Model(inputs = visible, outputs = output)
        
        self.model.compile(optimizer = opt,
                    loss = "binary_crossentropy",
                    metrics = ["accuracy"])
    
    def fit_CNN_model(self, batch_size = 128, epochs = 400, use_gpu = True):
        
        clr = CyclicLR(base_lr = 0.0001,
                max_lr = 0.01,
                step_size = 8,
                mode = "triangular2"
                )
        
        if use_gpu == True:
            
            gpu = self.find_GPU_name()
            
            tf.debugging.set_log_device_placement(True)
            
            with tf.device(gpu):
                
                start_time = time.time()
                
                self.model.fit(x = self.X_train,
                                y = self.y_train,
                                batch_size = batch_size,
                                epochs = epochs,
                                verbose = 0,
                                shuffle = True,
                                validation_data = (self.X_test, self.y_test),
                                callbacks = [clr])
                
                end_time = time.time()
        
        else:
    
            start_time = time.time()
            
            self.model.fit(x = self.X_train,
                            y = self.y_train,
                            batch_size = batch_size,
                            epochs = epochs,
                            verbose = 0,
                            shuffle = True,
                            validation_data = (self.X_test, self.y_test),
                            callbacks = [clr])
            
            end_time = time.time()
    
        print("Run Time: %.2f" % ((end_time - start_time)/60), " Minutes")
    
    def get_base_nn(self):
        
        weights = self.model.get_weights()
        CNSVM_weights = weights[:80]
        
        visible = Input(shape = (94,1))
        conv1 = Conv1D(64,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (visible)
        batch1 = BatchNormalization() (conv1)
        act1 = Activation(self.act) (batch1)
        conv2 = Conv1D(96,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (visible)
        batch2 = BatchNormalization() (conv2)
        act2 = Activation(self.act) (batch2)
        conv2_1 = Conv1D(128,
                        kernel_size = 3,
                        padding = "same",
                        kernel_initializer = "lecun_normal",
                        kernel_regularizer = l2(l = 0.0001)) (act2)
        batch2_1 = BatchNormalization() (conv2_1)
        act2_1 = Activation(self.act) (batch2_1)
        conv3 = Conv1D(16,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (visible)
        batch3 = BatchNormalization() (conv3)
        act3 = Activation(self.act) (batch3)
        conv3_1 = Conv1D(32,
                    kernel_size = 5,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (act3)
        batch3_1 = BatchNormalization() (conv3_1)
        act3_1 = Activation(self.act) (batch3_1)
        pool = MaxPooling1D(pool_size = 3, strides = 1, padding = "same") (visible)
        conv4 = Conv1D(32,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (pool) 
        batch4 = BatchNormalization() (conv4)
        act4 = Activation(self.act) (batch4)
        out = concatenate([act1, act2_1, act3_1, act4], axis = -1)
        conv5 = Conv1D(64,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (out)
        batch5 = BatchNormalization() (conv5)
        act5 = Activation(self.act) (batch5)
        conv6 = Conv1D(96,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (out)
        batch6 = BatchNormalization() (conv6)
        act6 = Activation(self.act) (batch6)
        conv6_1 = Conv1D(128,
                        kernel_size = 3,
                        padding = "same",
                        kernel_initializer = "lecun_normal",
                        kernel_regularizer = l2(l = 0.0001)) (act6)
        batch6_1 = BatchNormalization() (conv6_1)
        act6_1 = Activation(self.act) (batch6_1)
        conv7 = Conv1D(16,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (out)
        batch7 = BatchNormalization() (conv7)
        act7 = Activation(self.act) (batch7)
        conv7_1 = Conv1D(32,
                    kernel_size = 5,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (act7)
        batch7_1 = BatchNormalization() (conv7_1)
        act7_1 = Activation(self.act) (batch7_1)
        pool_2 = MaxPooling1D(pool_size = 3, strides = 1, padding = "same") (out)
        conv8 = Conv1D(32,
                    kernel_size = 1,
                    padding = "same",
                    kernel_initializer = "lecun_normal",
                    kernel_regularizer = l2(l = 0.0001)) (pool_2)
        batch8 = BatchNormalization() (conv8)
        act8 = Activation(self.act) (batch8)
        out2= concatenate([act5, act6_1, act7_1, act8], axis = -1)
        flat = Flatten() (out2)
        drop1 = Dropout(0.54) (flat)
        hidden1 = Dense(200, activation = self.act) (drop1)
        drop2 = Dropout(0.33) (hidden1)
        hidden2 = Dense(150, activation = self.act) (drop2)
        drop3 = Dropout(0.1) (hidden2)
        hidden3 = Dense(100, activation = self.act) (drop3)
        drop4 = Dropout(0.46) (hidden3)
        hidden4 = Dense(50, activation = self.act) (drop4)
        
        self.neural_model = Model(inputs = visible, outputs = hidden4)
        self.neural_model.set_weights(CNSVM_weights)

#

# train on VNIR dataset -----------------------------------------------------------------------------------

VNIR_model = CNSVM_VNIR(cost_sensitive = cost_sensitive)
VNIR_model.prepare_training_data(
    "VNIR_data.txt", ","
)
VNIR_model.define_model(
    select_act = activation_function,
    opt = optimization
)
VNIR_model.fit_CNN_model(
    use_gpu = True
)
VNIR_model.get_base_nn()

#

# transfer weights -----------------------------------------------------------------------------------

visible = Input(shape = (97,1))
conv1 = Conv1D(64,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (visible)
batch1 = BatchNormalization() (conv1)
act1 = Activation(activation_function) (batch1)
conv2 = Conv1D(96,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (visible)
batch2 = BatchNormalization() (conv2)
act2 = Activation(activation_function) (batch2)
conv2_1 = Conv1D(128,
                kernel_size = 3,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (act2)
batch2_1 = BatchNormalization() (conv2_1)
act2_1 = Activation(activation_function) (batch2_1)
conv3 = Conv1D(16,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (visible)
batch3 = BatchNormalization() (conv3)
act3 = Activation(activation_function) (batch3)
conv3_1 = Conv1D(32,
                kernel_size = 5,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (act3)
batch3_1 = BatchNormalization() (conv3_1)
act3_1 = Activation(activation_function) (batch3_1)
pool = MaxPooling1D(pool_size = 3, strides = 1, padding = "same") (visible)
conv4 = Conv1D(32,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (pool) 
batch4 = BatchNormalization() (conv4)
act4 = Activation(activation_function) (batch4)
out = concatenate([act1, act2_1, act3_1, act4], axis = -1)
conv5 = Conv1D(64,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (out)
batch5 = BatchNormalization() (conv5)
act5 = Activation(activation_function) (batch5)
conv6 = Conv1D(96,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (out)
batch6 = BatchNormalization() (conv6)
act6 = Activation(activation_function) (batch6)
conv6_1 = Conv1D(128,
                kernel_size = 3,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (act6)
batch6_1 = BatchNormalization() (conv6_1)
act6_1 = Activation(activation_function) (batch6_1)
conv7 = Conv1D(16,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (out)
batch7 = BatchNormalization() (conv7)
act7 = Activation(activation_function) (batch7)
conv7_1 = Conv1D(32,
                kernel_size = 5,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (act7)
batch7_1 = BatchNormalization() (conv7_1)
act7_1 = Activation(activation_function) (batch7_1)
pool_2 = MaxPooling1D(pool_size = 3, strides = 1, padding = "same") (out)
conv8 = Conv1D(32,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (pool_2)
batch8 = BatchNormalization() (conv8)
act8 = Activation(activation_function) (batch8)
out2= concatenate([act5, act6_1, act7_1, act8], axis = -1)
flat = Flatten() (out2)

drop1 = Dropout(0.54) (flat)
hidden1 = Dense(200, activation = activation_function) (drop1)

# define the frozen dense layer

drop2 = Dropout(0.33) (hidden1)
hidden2 = Dense(150, activation = activation_function, trainable = False) (drop2)
drop3 = Dropout(0.1) (hidden2)
hidden3 = Dense(100, activation = activation_function, trainable = False) (drop3)
drop4 = Dropout(0.46) (hidden3)
hidden4 = Dense(50, activation = activation_function, trainable = False) (drop4)

if cost_sensitive:
    output = Dense(1, activation = "sigmoid", bias_initializer = VNIR_model.bias_init) (hidden4)
else:
    output = Dense(1, activation = "sigmoid") (hidden4)

NIR_model = Model(inputs = visible, outputs = output)

# compile

NIR_model.compile(optimizer = optimization,
                   loss = "binary_crossentropy",
                   metrics = ["accuracy"])

# extract the pretrained model weights from the dense layer

VNIR_weights = VNIR_model.model.get_weights()[72:82]

# set only the weights of the frozen dense layers

# due to different convolutional layer sizes the first dense layer (n = 43) cannot be trained
# using transfer learning

NIR_model.layers[45].weights[0].assign(tf.convert_to_tensor(VNIR_weights[2]))
NIR_model.layers[45].weights[1].assign(tf.convert_to_tensor(VNIR_weights[3]))
NIR_model.layers[47].weights[0].assign(tf.convert_to_tensor(VNIR_weights[4]))
NIR_model.layers[47].weights[1].assign(tf.convert_to_tensor(VNIR_weights[5]))
NIR_model.layers[49].weights[0].assign(tf.convert_to_tensor(VNIR_weights[6]))
NIR_model.layers[49].weights[1].assign(tf.convert_to_tensor(VNIR_weights[7]))
NIR_model.layers[50].weights[0].assign(tf.convert_to_tensor(VNIR_weights[8]))
NIR_model.layers[50].weights[1].assign(tf.convert_to_tensor(VNIR_weights[9]))

#

# load NIR data -----------------------------------------------------------------------------------------------

NIR_dataset = np.loadtxt('NIR_data.txt', delimiter = ",")
X = NIR_dataset[:,0:97]
Y = NIR_dataset[:,97]

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.7,
                                                    random_state = 666)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#

# transfer learning -----------------------------------------------------------------------------------------------

clr = CyclicLR(base_lr = 0.0001,
                max_lr = 0.01,
                step_size = 8,
                mode = "triangular2"
                )

start_time = time.time()
            
training_performance = NIR_model.fit(x = X_train,
                                      y = y_train,
                                      batch_size = 128,
                                      epochs = 400,
                                      verbose = 0,
                                      shuffle = True,
                                      validation_data = (X_test, y_test),
                                      callbacks = [clr])

end_time = time.time()

print("Run Time: %.2f" % ((end_time - start_time)/60), " Minutes")

#

# train transfered SVM layer -----------------------------------------------------------------------------------------------

weights = NIR_model.get_weights()
CNSVM_weights = weights[:80]
visible = Input(shape = (97,1))
conv1 = Conv1D(64,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (visible)
batch1 = BatchNormalization() (conv1)
act1 = Activation(activation_function) (batch1)
conv2 = Conv1D(96,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (visible)
batch2 = BatchNormalization() (conv2)
act2 = Activation(activation_function) (batch2)
conv2_1 = Conv1D(128,
                kernel_size = 3,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (act2)
batch2_1 = BatchNormalization() (conv2_1)
act2_1 = Activation(activation_function) (batch2_1)
conv3 = Conv1D(16,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (visible)
batch3 = BatchNormalization() (conv3)
act3 = Activation(activation_function) (batch3)
conv3_1 = Conv1D(32,
                kernel_size = 5,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (act3)
batch3_1 = BatchNormalization() (conv3_1)
act3_1 = Activation(activation_function) (batch3_1)
pool = MaxPooling1D(pool_size = 3, strides = 1, padding = "same") (visible)
conv4 = Conv1D(32,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (pool) 
batch4 = BatchNormalization() (conv4)
act4 = Activation(activation_function) (batch4)
out = concatenate([act1, act2_1, act3_1, act4], axis = -1)
conv5 = Conv1D(64,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (out)
batch5 = BatchNormalization() (conv5)
act5 = Activation(activation_function) (batch5)
conv6 = Conv1D(96,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (out)
batch6 = BatchNormalization() (conv6)
act6 = Activation(activation_function) (batch6)
conv6_1 = Conv1D(128,
                kernel_size = 3,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (act6)
batch6_1 = BatchNormalization() (conv6_1)
act6_1 = Activation(activation_function) (batch6_1)
conv7 = Conv1D(16,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (out)
batch7 = BatchNormalization() (conv7)
act7 = Activation(activation_function) (batch7)
conv7_1 = Conv1D(32,
                kernel_size = 5,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (act7)
batch7_1 = BatchNormalization() (conv7_1)
act7_1 = Activation(activation_function) (batch7_1)
pool_2 = MaxPooling1D(pool_size = 3, strides = 1, padding = "same") (out)
conv8 = Conv1D(32,
                kernel_size = 1,
                padding = "same",
                kernel_initializer = "lecun_normal",
                kernel_regularizer = l2(l = 0.0001)) (pool_2)
batch8 = BatchNormalization() (conv8)
act8 = Activation(activation_function) (batch8)
out2= concatenate([act5, act6_1, act7_1, act8], axis = -1)
flat = Flatten() (out2)
drop1 = Dropout(0.54) (flat)
hidden1 = Dense(200, activation = activation_function) (drop1)
drop2 = Dropout(0.33) (hidden1)
hidden2 = Dense(150, activation = activation_function) (drop2)
drop3 = Dropout(0.1) (hidden2)
hidden3 = Dense(100, activation = activation_function) (drop3)
drop4 = Dropout(0.46) (hidden3)
hidden4 = Dense(50, activation = activation_function) (drop4)
neural_model = Model(inputs = visible, outputs = hidden4)
neural_model.set_weights(CNSVM_weights)

# extract features

X_neural_train = neural_model.predict(X_train)
X_neural_test = neural_model.predict(X_test)

# tune SVM activation layer

if cost_sensitive:
    SVM_class_weights = {0:class_weights[0], 1:class_weights[1]}
else:
    SVM_class_weights = None

tune_start = time.time()
space = {"C" : hp.uniform("C", 0, 500),
        "gamma" : hp.uniform("gamma", 0, 500)
        }
def objective(space):
        neural_svm = SVC(kernel = "rbf",
                        C = space["C"],
                        gamma = space["gamma"],
                        class_weight = SVM_class_weights
                        )
        accuracy = cross_val_score(neural_svm, X_neural_train, y_train, cv = 10).mean()
        return {"loss": -accuracy, "status": STATUS_OK }
trials = Trials()
best = fmin(fn = objective,
                space = space,
                algo = tpe.suggest,
                max_evals = 100,
                trials = trials
        )
tune_end = time.time()
print("Run Time: %.2f" % ((tune_end - tune_start)/60), " Minutes")

# train SVM activation layer

svm_start_time = time.time()
        
NIR_CNSVM = SVC(kernel = "rbf",
        C = best["C"],
        gamma = best["gamma"],
        probability = True,
        class_weight = SVM_class_weights
        ).fit(X_neural_train, y_train)

svm_end_time = time.time()
print("Run Time: %.2f" % ((svm_end_time- svm_start_time)/60), " Minutes")

#

# evaluate algorithm performance -----------------------------------------------------------------------------

predictions = NIR_CNSVM.predict(X_neural_test)
pred_probs = NIR_CNSVM.predict_proba(X_neural_test)
accuracy = classification_report(y_test, predictions)
Kappa = cohen_kappa_score(y_test, predictions)
confusion = confusion_matrix(y_test, predictions)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
sensitivity = (TP / float(TP + FN))
specificity = (TN / float(TN + FP))
roc_auc = roc_auc_score(y_test, NIR_CNSVM.predict(X_neural_test))
lr_probs = pred_probs[:, 1]
precision, recall, _ = precision_recall_curve(y_test, lr_probs)
pr_auc = auc(recall, precision)
mse = mse_calc(y_test, pred_probs)
print(accuracy)
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Kappa: {Kappa:.2f}")
print(f"SS AUC: {roc_auc:.2f}")
print(f"PR AUC: {pr_auc:.2f}")
print(f"MSE: {mse:.4f}")

#