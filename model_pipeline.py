import random

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error,binary_crossentropy,categorical_crossentropy
from keras.regularizers import L1
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve,auc as cal_auc
import time

start_time = time.time()

# val/train
val = True

def seed_tensorflow(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def softmax_with_t(t=1.0):
    def _softmat(x):
        x = x / t
        e_x = K.exp(x - K.max(x, axis=-1, keepdims=True))
        return e_x / K.sum(e_x, axis=-1, keepdims=True)
    return _softmat

os.chdir("/public3/home/sc72778/zgr/xmc")
seed_tensorflow()
# SC matrix
data = pd.read_csv("data/data.csv",index_col=0)
# Bulk matrix
bulk = pd.read_csv("data/bulk.csv",index_col=0)
# GSVA matrix
gsva_kegg = pd.read_csv("data/gsva-kegg.csv",index_col=0).T

kegg_names = gsva_kegg.columns
gene_names = bulk.columns

gsva_kegg = gsva_kegg.values


features,labels = data.iloc[:,:-1].values,data.values[:,-1]
labels = labels.astype(int)
bulk_samples = bulk.index
bulk = bulk.values

features = np.nan_to_num(features,nan=0)
bulk = np.nan_to_num(bulk,nan=0)

##############################################################################################################################
#............................................................................................................................#
##############################################################################################################################
# Pseudo label data generation

# Number of random samples
sample_count = 100
# Sampling times
sample_time = 1000

def data_generation(t=1):
    global pseudo_features,pseudo_labels
    pseudo_features = []
    pseudo_labels = []
    # Use numpy.bincount to obtain the number of occurrences of each element
    counts = np.bincount(labels)
    ws = 1/counts
    w_vec = np.ones_like(labels).astype("float64")
    for i,w in enumerate(ws):
        w_vec[labels==i] *= w

    w_vec = w_vec/w_vec.sum()
    # Imbalanced variable
    unbalanced_vars = []
    for i,w in enumerate(ws):
        unbalanced_var = w_vec.copy()
        unbalanced_var[labels==i] *= 2
        unbalanced_var /= unbalanced_var.sum()
        unbalanced_vars.append(unbalanced_var)

    # train: 0, val: sample_time * t
    if val:
        r_ = sample_time * t
    else:
        r_ = 0

    for i_ in tqdm(range(sample_time)):
        np.random.seed(i_ + r_)
        p = unbalanced_vars[np.random.randint(0,len(unbalanced_vars))]
        samples = np.random.choice(np.arange(len(labels)),sample_count,p=p,replace=False)
        mean_feature = features[samples].sum(axis=0)
        pseudo_features.append(mean_feature)
        counts = np.bincount(labels[samples])
        pseudo_label = np.argmax(counts)
        pseudo_labels.append(pseudo_label)

    pseudo_features,pseudo_labels = np.array(pseudo_features),np.array(pseudo_labels)

    pseudo_labels = tf.one_hot(pseudo_labels, 6).numpy()

    # Both bulk and single cells undergo log2 conversion
    pseudo_features = np.log2(pseudo_features + 1)
    pseudo_features = minmax_scale(pseudo_features)

data_generation(t=1)

# Bulk sampling
bulk_expands = []
indexs = np.arange(len(bulk))
for i in tqdm(range(sample_time)):
    np.random.seed(i)
    i_ = i % len(bulk)
    if i_ == 0:
        np.random.shuffle(indexs)
    sample = bulk[indexs[i_]]
    bulk_expands.append(sample)

bulk_expands = np.array(bulk_expands)

bulk_expands = minmax_scale(bulk_expands)

##############################################################################################################################
#............................................................................................................................#
##############################################################################################################################
# Pre-trained model (domain adaptive model)

# Hyper-parameters
learning_rate = 1e-3
epochs = 400
batch_size = 100

class AE(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Dense(256,activation="sigmoid")
        self.decoder = Dense(bulk.shape[1], activation="sigmoid")

    def call(self, inputs):
        encode = self.encoder(inputs)
        decode = self.decoder(encode)
        return encode,decode
    
    def predict(self, x):
        encode = self.encoder(x)
        return encode

class CE(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_1 = Dense(256,activation="sigmoid")
        self.layer_2 = Dense(6, activation=softmax_with_t(t=30))

    def call(self, inputs):
        o_1 = self.layer_1(inputs)
        output = self.layer_2(o_1)
        return o_1,output
    
    def predict(self, x):
        output = self.layer_2(x)
        return output

class DomainAdaptation(Model):
    def __init__(self,AE_model,CE_model):
        super(DomainAdaptation, self).__init__()
        self.AE_model = AE_model
        self.CE_model = CE_model
    
    def compile(self,optimizer):
        super(DomainAdaptation, self).compile(optimizer=optimizer)

    def train_step(self, data):
        # Unpack data
        (bulk,pseudo_features), pseudo_labels = data
        with tf.GradientTape() as tape:
            # Forward pass
            encode,decode = self.AE_model(bulk)
            o_1,output = self.CE_model(pseudo_features)
            # Compute losses
            category_loss = 0.1 * categorical_crossentropy(pseudo_labels,output)
            recon_loss = 0.1 * mean_squared_error(bulk,decode)
            mmd_loss = 0.8 * self.compute_mmd(o_1,encode)
            loss = category_loss + recon_loss + mmd_loss

        # Compute gradients
        trainable_vars = self.AE_model.trainable_variables
        trainable_vars.extend(self.CE_model.trainable_variables)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(pseudo_labels, output)
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "loss":tf.reduce_mean(loss),
            "category_loss":tf.reduce_mean(category_loss),
            "recon_loss":tf.reduce_mean(recon_loss),
            "mmd_loss":tf.reduce_mean(mmd_loss)
        })
        return results
    
    @staticmethod
    def kl_divergence(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

    @staticmethod
    def compute_kernel(x, y):
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
        tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
        return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

# Domain adaptive model
seed_tensorflow()
AE_model = AE()
CE_model = CE()
AE_model_path = "output/bulk/model/AE_model_path.h5"
CE_model_path = "output/bulk/model/CE_model_path.h5"
AE_model.build((None,bulk_expands.shape[1]))
CE_model.build((None,pseudo_features.shape[1]))
AE_model.load_weights(AE_model_path)
CE_model.load_weights(CE_model_path)

##############################################################################################################################
#............................................................................................................................#
##############################################################################################################################
# A Response Based Online Knowledge Distillation Model

encode = AE_model.predict(bulk).numpy()
scores_base = CE_model.predict(encode).numpy()

teacher_risk_scores = minmax_scale(scores_base[:,5])

# Hyper-parameters
learning_rate = 1e-3
epochs = 4000
batch_size = 100

indexs = np.arange(len(bulk))

scores_mean = []
weights_mean = []
for i in range(4):
    seed_tensorflow(seed=i)
    np.random.shuffle(indexs)
    # Teacher model (pre-trained)
    teacher_scores = teacher_risk_scores[indexs]

    # Student model
    input = Input(gsva_kegg.shape[1:])
    o1 = Dense(128, kernel_regularizer=L1(1e-4), activation="relu")(input)
    output = Dense(1, activation="sigmoid")(o1)
    model = Model(input,output)
    model.compile(optimizer=Adam(learning_rate), loss=binary_crossentropy, metrics=["mse"])
    model.fit(
        x=gsva_kegg[indexs],
        y=teacher_scores,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Student model prediction
    scores = model.predict(gsva_kegg)
    scores_mean.append(scores)

    weights = model.layers[1].get_weights()[0]
    weights = np.nan_to_num(weights, nan=0)
    weights = np.sum(np.square(weights), 1)
    weights_mean.append(weights)

end_time = time.time()

print("runtime:", end_time - start_time, "s")
print("# # # # # # #")