import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout,Embedding,SpatialDropout1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import math

from scripts import utils


class HyperparameterOptimisation:

    def __init__(self):
        """ Init method. """

    def train_model(self, config, reverse_dictionary, train_data, train_labels, test_data, test_labels, class_weights):
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
        best_model_params = dict()
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=1e-1,patience=3, restore_best_weights=True)

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

        def nDCG(y_true, y_pred):
            val,ind=tf.math.top_k(y_pred,k=16)

            y_pred_values=val.numpy()
            y_pred_location=ind.numpy()

            val,ind=tf.math.top_k(y_true,k=16)
            y_acc_location=ind.numpy()
            y_acc_values=val.numpy()
            
            
            mean=0
            for indexx,example in enumerate(y_pred_location):
                dcg=0
                for index,i in enumerate(y_pred_location[indexx]):
                    if (i in y_acc_location[indexx]):
                        if (y_pred_values[indexx][index]>0.5):
                            dcg+=2/math.log(i+2)
                        else:
                            dcg+=1/math.log(i+2)
                idcg=0
                for i in y_acc_values[indexx]:
                    if (i>0.5):
                        idcg+=2/math.log(i+2)
                mean+=dcg/idcg
            mean/=len(y_pred_location)    
            return tf.convert_to_tensor(mean,dtype=tf.float32)

        def Recall(y_true, y_pred):
            val,ind=tf.math.top_k(y_pred,k=16)

            y_pred_values=val.numpy()
            y_pred_location=ind.numpy()
            val,ind=tf.math.top_k(y_true,k=16)
            y_acc_location=ind.numpy()
            
            mean=0
            for indexx,example in enumerate(y_pred_location):
                true_positive=0
                truee=0
                for index,i in enumerate(y_pred_location[indexx]):
                    if (i in y_acc_location[indexx]):
                        true_positive+=y_pred_values[indexx][index]
                        truee+=1
                if (truee>0):
                    mean+=true_positive/truee
            mean/=len(y_pred_location)    
            return tf.convert_to_tensor(mean,dtype=tf.float32)

        def Precision(y_true, y_pred):
            val,ind=tf.math.top_k(y_pred,k=16)

            y_pred_values=val.numpy()
            y_pred_location=ind.numpy()

            val,ind=tf.math.top_k(y_true,k=16)
            y_acc_location=ind.numpy()
            y_acc_values=val.numpy()
            
            
            mean=0
            for indexx,example in enumerate(y_pred_location):
                true_positive=0
                truee=0
                for index,i in enumerate(y_pred_location[indexx]):
                    if (i in y_acc_location[indexx]):
                        true_positive+=y_pred_values[indexx][index]
                
                for i in y_acc_values[indexx]:
                    if (i>0.5):
                        truee=truee+1
                if (truee>0):
                    mean+=true_positive/truee
            mean/=len(y_pred_location)    
            return tf.convert_to_tensor(mean,dtype=tf.float32)
        
        def f1(y_true, y_pred):
            precision = Precision(y_true, y_pred)
            recall = Recall(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))
        
                     
        def create_model(params):
            model = Sequential()
            model.add(Embedding(dimensions, int(params["embedding_size"]), mask_zero=True))
            model.add(SpatialDropout1D(params["spatial_dropout"]))
            model.add(GRU(int(params["units"]), dropout=params["dropout"], recurrent_dropout=0, return_sequences=True, activation="tanh"))
            model.add(Dropout(params["dropout"]))
            model.add(GRU(int(params["units"]), dropout=params["dropout"], recurrent_dropout=0, return_sequences=False, activation="tanh"))
            model.add(Dropout(params["dropout"]))
            model.add(Dense(2 * dimensions, activation="sigmoid"))
            optimizer_rms = RMSprop(lr=params["learning_rate"])
            batch_size = int(params["batch_size"])
            model.compile(loss=utils.weighted_loss(class_weights), optimizer=optimizer_rms,metrics=[tf.keras.metrics.BinaryAccuracy(),nDCG,Precision,Recall,f1])
            model_fit = model.fit(
                train_data,
                train_labels,
                steps_per_epoch=len(train_data) // batch_size,
                epochs=optimize_n_epochs,
                callbacks=[early_stopping],
                validation_data=(test_data, test_labels),
                verbose=2,
                shuffle=True
            )
            return {'loss': model_fit.history["val_loss"][-1], 'status': STATUS_OK, 'model': model,"history":model_fit.history}
        # minimize the objective function using the set of parameters above
        trials = Trials()
        learned_params = fmin(create_model, params, trials=trials, algo=tpe.suggest, max_evals=int(config["max_evals"]))
        best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
        histories=[]
        for i in trials.results:
            histories.append(i['history'])
        utils.write_file("data/evaluationHistory.json", histories)
        
        # set the best params with respective values
        for item in learned_params:
            item_val = learned_params[item]
            best_model_params[item] = item_val
        return best_model_params, best_model
