import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Activation, Dense, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import time
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, cohen_kappa_score, mean_squared_error
from sklearn.metrics import precision_recall_curve, auc
from sklearn.svm import *
import hyperopt
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from joblib import dump, load
import pandas as pd

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

class CNSVM_Model:
    
    X = 0
    Y = 0
    
    def __init__(self, seed = 666):
        self.seed = seed
        np.random.seed(self.seed)
    
    def prepare_training_data(self, dataset, test_size = 0.3):
        
        self.X = dataset[:,1:98]
        self.Y = dataset[:,0]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y,
                                                                                test_size = test_size,
                                                                                random_state = self.seed)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
    
    def find_GPU_name(self):
        gpu_name = tf.config.experimental.list_logical_devices("GPU")
        return(gpu_name[0])
    
    def define_model(self, select_act = "relu", opt = "sgd", bias_init = None):
        
        if select_act == "relu":
            self.act = "relu"
        elif select_act == "swish":
            self.act = "swish"
        else:
            return(print("Select a valid activation function between ReLU and Swish"))
        
        if bias_init is not None:
            tf.keras.initializers.Constant(bias_init)
        
        visible = Input(shape = (97,1))
        
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
                                                            
        # Fully connected dense layers
        
        flat = Flatten() (out2)
        drop1 = Dropout(0.54) (flat)
        hidden1 = Dense(200, activation = self.act) (drop1)
        drop2 = Dropout(0.33) (hidden1)
        hidden2 = Dense(150, activation = self.act) (drop2)
        drop3 = Dropout(0.1) (hidden2)
        hidden3 = Dense(100, activation = self.act) (drop3)
        drop4 = Dropout(0.46) (hidden3)
        hidden4 = Dense(50, activation = self.act) (drop4)
        output = Dense(1, activation = "sigmoid", bias_initializer = bias_init) (hidden4)
        
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
        
        # Redefine model without final layer
        
        visible = Input(shape = (97,1))
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
        
        # Set weights
        
        self.neural_model = Model(inputs = visible, outputs = hidden4)
        self.neural_model.set_weights(CNSVM_weights)
    
    def tune_SVM(self, class_weights = None):
        
        if class_weights is not None:
            self.SVM_class_weights = {0:class_weights[0], 1:class_weights[1]}
        else:
            self.SVM_class_weights = None
        
        self.X_neural_train = self.neural_model.predict(self.X_train)
        self.X_neural_test = self.neural_model.predict(self.X_test)
        
        tune_start = time.time()
        space = {"C" : hp.uniform("C", 0, 500),
                "gamma" : hp.uniform("gamma", 0, 500)
                }
        def objective(space):
            neural_svm = SVC(kernel = "rbf",
                            C = space["C"],
                            gamma = space["gamma"],
                            class_weight = self.SVM_class_weights
                            )
            accuracy = cross_val_score(neural_svm, self.X_neural_train, self.y_train, cv = 10).mean()
            return {"loss": -accuracy, "status": STATUS_OK }
        trials = Trials()
        self.best = fmin(fn = objective,
                    space = space,
                    algo = tpe.suggest,
                    max_evals = 100,
                    trials = trials
                )
        tune_end = time.time()
        print("Run Time: %.2f" % ((tune_end - tune_start)/60), " Minutes")
     
    def fit_SVM(self):
        
        svm_start_time = time.time()
        
        self.NSVM = SVC(kernel = "rbf",
                C = self.best["C"],
                gamma = self.best["gamma"],
                probability = True,
                class_weight = self.SVM_class_weights
                ).fit(self.X_neural_train, self.y_train)
        
        svm_end_time = time.time()
        
        print("Run Time: %.2f" % ((svm_end_time - svm_start_time)/60), " Minutes")
    
    def evaluate_CNSVM(self):
        predictions = self.NSVM.predict(self.X_neural_test)
        pred_probs = self.NSVM.predict_proba(self.X_neural_test)
        accuracy = classification_report(self.y_test, predictions)
        Kappa = cohen_kappa_score(self.y_test, predictions)
        confusion = confusion_matrix(self.y_test, predictions)
        TP = confusion[1,1]
        TN = confusion[0,0]
        FP = confusion[0,1]
        FN = confusion[1,0]
        sensitivity = (TP / float(TP + FN))
        specificity = (TN / float(TN + FP))
        roc_auc = roc_auc_score(self.y_test, self.NSVM.predict(self.X_neural_test))
        lr_probs = pred_probs[:, 1]
        self.precision, self.recall, _ = precision_recall_curve(self.y_test, lr_probs)
        pr_auc = auc(self.recall, self.precision)
        mse = mse_calc(self.y_test, pred_probs)
        print(accuracy)
        print(f"Sensitivity: {sensitivity:.2f}")
        print(f"Specificity: {specificity:.2f}")
        print(f"Kappa: {Kappa:.2f}")
        print(f"SS AUC: {roc_auc:.2f}")
        print(f"PR AUC: {pr_auc:.2f}")
        print(f"MSE: {mse:.4f}")
    
    def save_model(self, target_file_name):
        self.neural_model.save(target_file_name + "_Neural_Net.h5")
        dump(self.NSVM, target_file_name + "_SVM_Actiavtion_Layer.joblib")
    
    def load_model(self, neural_model, activation_layer):
        self.neural_model = load_model(neural_model)
        self.NSVM = load(activation_layer)
    
    def predict_label(self, target_data):
        target_data = target_data.reshape(1, 97, 1)
        extracted_features = self.neural_model.predict(target_data)
        labels = self.NSVM.predict(extracted_features)
        return(labels)
    
    def predict_prob(self, target_data):
        target_data = target_data.reshape(1, 97, 1)
        extracted_features = self.neural_model.predict(target_data)
        probs = self.NSVM.predict_proba(extracted_features)
        return(probs)

class CNSVM_Model_categorical(CNSVM_Model):
    
    def prepare_training_data(self, dataset, test_size = 0.3):
        
        self.X = dataset.values[:,1:98].astype(float)
        self.Y = dataset.iloc[:,0].astype("category")
        
        self.encoder = LabelEncoder()
        self.encoder.fit(self.Y)
        self.Y_encoded = self.encoder.transform(self.Y)
        self.dummy_Y =  tf.keras.utils.to_categorical(self.Y_encoded)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y,
                                                                                test_size = test_size,
                                                                                random_state = self.seed)
        
        
        self.train_encoded_y = self.encoder.transform(self.y_train)
        self.test_encoded_y = self.encoder.transform(self.y_test)
        self.dummy_y_train = tf.keras.utils.to_categorical(self.train_encoded_y) # One Hot Encoded - Convert Integers to Dummy 
        self.dummy_y_test = tf.keras.utils.to_categorical(self.test_encoded_y)   
            
    
        self.key = pd.DataFrame(index = range(self.Y.cat.categories.size), 
                            columns = ["key_value", "sample"])
        for i in range(self.Y.cat.categories.size):
            self.key.loc[i, "key_value"] = i
            self.key.loc[i, "sample"] = self.Y.cat.categories[i]
        print(self.key)
        
    def define_model(self, select_act = "relu", opt = "sgd"):
        
        if select_act == "relu":
            self.act = "relu"
        elif select_act == "swish":
            self.act = "swish"
        else:
            return(print("Select a valid activation function between ReLU and Swish"))
        
        visible = Input(shape = (97,1))
        
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
                                                            
        # Fully connected dense layers
        
        flat = Flatten() (out2)
        drop1 = Dropout(0.54) (flat)
        hidden1 = Dense(200, activation = self.act) (drop1)
        drop2 = Dropout(0.33) (hidden1)
        hidden2 = Dense(150, activation = self.act) (drop2)
        drop3 = Dropout(0.1) (hidden2)
        hidden3 = Dense(100, activation = self.act) (drop3)
        drop4 = Dropout(0.46) (hidden3)
        hidden4 = Dense(50, activation = self.act) (drop4)
        output = Dense(self.key.shape[0], activation = "softmax") (hidden4)
        
        self.model = Model(inputs = visible, outputs = output)
        
        # compilation
        
        self.model.compile(optimizer = opt,
                            loss = "categorical_crossentropy",
                            metrics = ["categorical_accuracy"])

    def fit_CNN_model(self, batch_size = 128, epochs = 400, use_gpu = True):
        
        # Cyclic Learning Rate
        
        clr = CyclicLR(base_lr = 0.0001,
                max_lr = 0.01,
                step_size = 8,
                mode = "triangular2"
                )

        # Train Convolutional Neural Network
        
        if use_gpu == True:
            
            gpu = self.find_GPU_name()
            
            tf.debugging.set_log_device_placement(True)
            
            with tf.device(gpu):
                
                start_time = time.time()
                
                self.model.fit(x = self.X_train,
                                y = self.dummy_y_train,
                                batch_size = batch_size,
                                epochs = epochs,
                                verbose = 0,
                                shuffle = True,
                                validation_data = (self.X_test, self.dummy_y_test),
                                callbacks = [clr])
                
                end_time = time.time()
        
        else:
    
            start_time = time.time()
            
            self.model.fit(x = self.X_train,
                            y = self.dummy_y_train,
                            batch_size = batch_size,
                            epochs = epochs,
                            verbose = 0,
                            shuffle = True,
                            validation_data = (self.X_test, self.dummy_y_test),
                            callbacks = [clr])
            
            end_time = time.time()
    
        print("Run Time: %.2f" % ((end_time - start_time)/60), " Minutes")

    def tune_SVM(self, c_limit = 500, gamma_limit = 500):
        
        # Perform feature extraction
        
        self.X_neural_train = self.neural_model.predict(self.X_train)
        self.X_neural_test = self.neural_model.predict(self.X_test)
        
        # Tune SVM on extracted features #con estadistica bayesiana se calcula el cost y gamma
        
        tune_start = time.time()
        space = {"C" : hp.uniform("C", 0, c_limit),
                "gamma" : hp.uniform("gamma", 0, gamma_limit)
                }
        def objective(space):
            neural_svm = SVC(kernel = "rbf",
                            C = space["C"],
                            gamma = space["gamma"]
                            )
            accuracy = cross_val_score(neural_svm, self.X_neural_train, self.train_encoded_y, cv = 10).mean()
            return {"loss": -accuracy, "status": STATUS_OK }
        trials = Trials()
        self.best = fmin(fn = objective,
                    space = space,
                    algo = tpe.suggest,
                    max_evals = 100,
                    trials = trials
                )
        tune_end = time.time()
        print("Run Time: %.2f" % ((tune_end - tune_start)/60), " Minutes")
    
    def fit_SVM(self):
        
        # Fit SVM on the tuned hyperparameters
        
        svm_start_time = time.time()
        
        self.NSVM = SVC(kernel = "rbf",
                C = self.best["C"],
                gamma = self.best["gamma"],
                probability = True,
                ).fit(self.X_neural_train, self.train_encoded_y)
        
        svm_end_time = time.time()
        
        print("Run Time: %.2f" % ((svm_end_time - svm_start_time)/60), " Minutes")

    def evaluate_CNSVM(self):
        predictions = self.NSVM.predict(self.X_neural_test)
        pred_probs = self.NSVM.predict_proba(self.X_neural_test)
        accuracy = classification_report(self.test_encoded_y, predictions)
        Kappa = cohen_kappa_score(self.test_encoded_y, predictions)
        confusion = confusion_matrix(self.test_encoded_y, predictions)
        print(accuracy)
        print(f"Kappa: {Kappa:.2f}")
        print(confusion)
        print(pred_probs)

if __name__ == "__main__":
    print("\nImported Package Versions:\n\n-------------------------------\n")
    print(f"Tensorflow: {tf.__version__}")
    print(f"Numpy: {np.__version__}")
    print(f"Scikit Learn: {sklearn.__version__}")
    print(f"Hyperopt: {hyperopt.__version__}")
    print(f"\n")
