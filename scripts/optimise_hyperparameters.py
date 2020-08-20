"""
Find the optimal combination of hyperparameters
"""

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

from scripts import layers
from scripts import utils


class HyperparameterOptimisation:

    def __init__(self):
        """ Init method. """

    def train_model(self, config, reverse_dictionary, train_data, train_labels, test_data, test_labels, tool_tr_samples, class_weights):
        """
        Train a model and report accuracy
        """
        # convert items to integer
        l_batch_size = list(map(int, config["batch_size"].split(",")))
        l_embedding_size = list(map(int, config["embedding_size"].split(",")))
        l_units = list(map(int, config["units"].split(",")))

        # convert items to float
        l_learning_rate = list(map(float, config["learning_rate"].split(",")))
        l_dropout = list(map(float, config["dropout"].split(",")))
        l_spatial_dropout = list(map(float, config["spatial_dropout"].split(",")))
        l_recurrent_dropout = list(map(float, config["recurrent_dropout"].split(",")))

        optimize_n_epochs = int(config["optimize_n_epochs"])

        # get dimensions
        dimensions = len(reverse_dictionary) + 1
        max_len = train_data.shape[1]
        
        best_model_params = dict()
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=1e-1, restore_best_weights=True)

        # specify the search space for finding the best combination of parameters using Bayesian optimisation
        params = {
            "embedding_size": hp.quniform("embedding_size", l_embedding_size[0], l_embedding_size[1], 1),
            "units": hp.quniform("units", l_units[0], l_units[1], 1),
            "batch_size": hp.quniform("batch_size", l_batch_size[0], l_batch_size[1], 1),
            "learning_rate": hp.loguniform("learning_rate", np.log(l_learning_rate[0]), np.log(l_learning_rate[1])),
            "dropout": hp.uniform("dropout", l_dropout[0], l_dropout[1]),
            "spatial_dropout": hp.uniform("spatial_dropout", l_spatial_dropout[0], l_spatial_dropout[1]),
            "recurrent_dropout": hp.uniform("recurrent_dropout", l_recurrent_dropout[0], l_recurrent_dropout[1])
        }
        
        def create_model(params):

            embedding_size = int(params["embedding_size"])
            gru_units = int(params["units"])
            spatial1d_dropout = float(params["spatial_dropout"])
            dropout = float(params["dropout"])
            recurrent_dropout = float(params["recurrent_dropout"])
            learning_rate = params["learning_rate"]
            batch_size = int(params["batch_size"])

            sequence_input = tf.keras.layers.Input(shape=(max_len,), dtype='int32')
            embedded_sequences = tf.keras.layers.Embedding(dimensions, embedding_size, input_length=max_len, mask_zero=True)(sequence_input)
            
            embedded_sequences = tf.keras.layers.SpatialDropout1D(spatial1d_dropout)(embedded_sequences)

            gru1_output = tf.keras.layers.GRU(
                gru_units,
                return_sequences=True,
                return_state=False,
                activation='elu',
                recurrent_dropout=recurrent_dropout
            )(embedded_sequences)

            gru1_dropout = tf.keras.layers.Dropout(dropout)(gru1_output)

            gru2_output, gru2_hidden = tf.keras.layers.GRU(
                gru_units,
                return_sequences=True,
                return_state=True,
                activation='elu',
                recurrent_dropout=recurrent_dropout
            )(gru1_dropout)

            gru2_dropout  = tf.keras.layers.Dropout(dropout)(gru2_hidden)

            output = tf.keras.layers.Dense(2 * dimensions, activation='sigmoid')(gru2_dropout)

            model = tf.keras.Model(inputs=sequence_input, outputs=output)
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                loss=utils.weighted_loss(class_weights),
            )

            print(model.summary())
            model_fit = model.fit(
                utils.balanced_sample_generator(
                    train_data,
                    train_labels,
                    batch_size,
                    tool_tr_samples,
                    reverse_dictionary
                ),
                steps_per_epoch=len(train_data) // batch_size,
                epochs=optimize_n_epochs,
                callbacks=[early_stopping],
                validation_data=(test_data, test_labels),
                verbose=2,
                shuffle=True
            )
            return {'loss': model_fit.history["val_loss"][-1], 'status': STATUS_OK, 'model': model}
        # minimize the objective function using the set of parameters above
        trials = Trials()
        learned_params = fmin(create_model, params, trials=trials, algo=tpe.suggest, max_evals=int(config["max_evals"]))
        best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
        # set the best params with respective values
        for item in learned_params:
            item_val = learned_params[item]
            best_model_params[item] = item_val
        return best_model_params, best_model
