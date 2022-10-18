from typing import Union, Dict,List

from fastapi import FastAPI
from pydantic import BaseModel


import numpy as np
import json
import warnings
import operator

import h5py
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K



app = FastAPI()

warnings.filterwarnings("ignore")


def read_file(file_path):
    with open(file_path, 'r') as data_file:
        data = json.loads(data_file.read())
    return data


def create_model(model_path):
    reverse_dictionary = dict((str(v), k) for k, v in dictionary.items())
    model_weights = list()
    weight_ctr = 0
    cu_wt = list()
    for index, item in enumerate(trained_model.keys()):
        if "weight_" in item:
            d_key = "weight_" + str(weight_ctr)
            weights = trained_model.get(d_key)[()]
            mean = np.mean(weights)
            cu_wt.append(mean)
            model_weights.append(weights)
            weight_ctr += 1
    print("Overall mean of model weights: %.6f" % np.mean(cu_wt))
    # set the model weights
    loaded_model.set_weights(model_weights)
    return loaded_model, dictionary, reverse_dictionary

def get_predicted_tools(base_tools, predictions, topk):
    """
    Get predicted tools. If predicted tools are less in number, combine them with published tools
    """
    intersection = list(set(predictions).intersection(set(base_tools)))
    return intersection[:topk]

def sort_by_usage(t_list, class_weights,predictions, d_dict):
    """
    Sort predictions by usage/class weights
    """
    tool_dict = dict()
    for tool in t_list:
        t_id = d_dict[tool]
        tool_dict[tool] = predictions[t_id]
    tool_dict = dict(sorted(tool_dict.items(), key=lambda kv: kv[1], reverse=True))
    return list(tool_dict.keys()), list(tool_dict.values())

def separate_predictions(prediction, last_tool_name, weight_values, topk):
    """
    Get predictions from published and normal workflows
    """
    last_base_tools = list()
    predictions = prediction * weight_values
    prediction_pos = np.argsort(predictions, axis=-1)
    topk_prediction_pos = prediction_pos[-topk:]
    
    # get tool ids
    sorted_c_v={reverse_dictionary[str(tool_pos)]:predictions[tool_pos] for tool_pos in topk_prediction_pos}
    return sorted_c_v

def compute_recommendations(model, tool_sequence, labels, dictionary, reverse_dictionary, class_weights,model_rec, tool_sequence_rec, labels_rec, dictionary_rec, reverse_dictionary_rec, class_weights_rec, topk=10, max_seq_len=25):
    tl_seq = tool_sequence
    tl_seq_ids = [str(dictionary[t]) for t in tl_seq]
    last_tool_name = tl_seq[-1]
    toPredict=tool_sequence_rec[-1]
    sample = np.zeros(max_seq_len)
    weight_val = list(class_weights.values())
    weight_val = np.reshape(weight_val, (len(weight_val),))
    weight_val_rec = list(class_weights_rec.values())
    weight_val_rec = np.reshape(weight_val_rec, (len(weight_val_rec),))
    for idx, tool_id in enumerate(tl_seq_ids):
        sample[idx] = int(tool_id)
    sample_reshaped = np.reshape(sample, (1, max_seq_len))
    # predict next tools for a test path
    tl_seq_rec = tool_sequence_rec
    tl_seq_ids_rec = [str(dictionary[t]) for t in tl_seq_rec]
    sample_rec = np.zeros(max_seq_len)
    for idx, tool_id in enumerate(tl_seq_ids_rec):
        sample_rec[idx] = int(tool_id)
    sample_reshaped_rec = np.reshape(sample_rec, (1, max_seq_len))
    predictionLeft = model.predict(sample_reshaped, verbose=0)
    predictionRight = model_rec.predict(sample_reshaped_rec, verbose=0)
    
    nw_dimension = predictionLeft.shape[1]
    predictionLeft = np.reshape(predictionLeft, (nw_dimension,))
    
    nw_dimension_rec = predictionRight.shape[1]
    predictionRight = np.reshape(predictionRight, (nw_dimension,))
    half_len = int(nw_dimension / 2)
    half_len_rec = int(nw_dimension_rec / 2)
    
    pub_t = separate_predictions(predictionLeft[:half_len], last_tool_name, weight_val, topk)
    pub_t_rec = separate_predictions(predictionRight[:half_len_rec], last_tool_name, weight_val_rec, topk)

    # remove duplicates if any
    intop=0
    predictList=[]
    if (toPredict in pub_t) or (last_tool_name in pub_t_rec):
        predictList.append(({'node':[''],
            'connect':{toPredict:[last_tool_name]}},10000))
        if (toPredict in pub_t):
            pub_t.pop(toPredict)
        if (last_tool_name in pub_t_rec):
            pub_t_rec.pop(last_tool_name)
    tempOneElementList=[]
    for i in pub_t:
        if i in pub_t_rec:
            predictList.append(({'node':[i],
            'connect':{toPredict:[i],i:[last_tool_name]}},1000))
            tempOneElementList.append(i)
    for i in tempOneElementList:
        pub_t.pop(i)
    for i in pub_t:
        predictList.append(({'node':[i],
            'connect':{i:[last_tool_name]}},pub_t[i]))
    for i in pub_t_rec:
        predictList.append(({'node':[i],
            'connect':{toPredict:[i]}},pub_t_rec[i]))
            
    return predictList


def compute_recommendations_OneDirection(model, tool_sequence, labels, dictionary, reverse_dictionary, class_weights, topk=10, max_seq_len=25):
    tl_seq = tool_sequence
    tl_seq_ids = [str(dictionary[t]) for t in tl_seq]
    last_tool_name = tl_seq[-1]
    sample = np.zeros(max_seq_len)
    weight_val = list(class_weights.values())
    weight_val = np.reshape(weight_val, (len(weight_val),))
    for idx, tool_id in enumerate(tl_seq_ids):
        sample[idx] = int(tool_id)
    sample_reshaped = np.reshape(sample, (1, max_seq_len))
    # predict next tools for a test path
    predictionLeft = model.predict(sample_reshaped, verbose=0)
    
    nw_dimension = predictionLeft.shape[1]
    predictionLeft = np.reshape(predictionLeft, (nw_dimension,))
    half_len = int(nw_dimension / 2)
    
    pub_t = separate_predictions(predictionLeft[:half_len], last_tool_name, weight_val, topk)       
    return pub_t



# load Model
model_path = "data/tool_recommendation_model_20_05.hdf5"
trained_model = h5py.File(model_path, 'r')
model_config = json.loads(trained_model.get('model_config')[()])
dictionary = json.loads(trained_model.get('data_dictionary')[()])
class_weights = json.loads(trained_model.get('class_weights')[()])
standard_connections = json.loads(trained_model.get('standard_connections')[()])
compatible_tools = json.loads(trained_model.get('compatible_tools')[()])
loaded_model = model_from_json(model_config)
model, dictionary, reverse_dictionary = create_model(model_path)
model_path_rec = "data/tool_recommendation_model_rec.hdf5"
trained_model_rec = h5py.File(model_path_rec, 'r')
model_config_rec = json.loads(trained_model_rec.get('model_config')[()])
dictionary_rec = json.loads(trained_model_rec.get('data_dictionary')[()])
class_weights_rec = json.loads(trained_model_rec.get('class_weights')[()])
standard_connections_rec = json.loads(trained_model_rec.get('standard_connections')[()])
compatible_tools_rec = json.loads(trained_model_rec.get('compatible_tools')[()])
loaded_model_rec = model_from_json(model_config_rec)
model_rec, dictionary_rec, reverse_dictionary_rec = create_model(model_path_rec)


def getAllPaths(parent_graph,node):
    graph=[]
    if node in parent_graph:
        for i in parent_graph[node]:
            tempGraph=getAllPaths(parent_graph,i)
            for j in tempGraph:
                k=j
                k.append(node)
                graph.append(k)
    if len(graph)==0:
        graph=[[node]]
    return graph
def parentGraphToChildGraph(parent_graph):
    child_graph={}
    for i in parent_graph:
        for j in parent_graph[i]:
            if j in child_graph:
                child_graph[j].append(i)
            else:
                child_graph[j]=[i]
    return child_graph

def newAlgo(to_predict_parent_graph,focused_elements,direction='right'):
    childGraph=parentGraphToChildGraph(to_predict_parent_graph)
    predictions=[]
    if (len(focused_elements)==2):
        previousPaths=getAllPaths(to_predict_parent_graph,focused_elements[0])
        succPaths=getAllPaths(childGraph,focused_elements[1])
        
        for i in previousPaths:
            for j in succPaths:
                predictions.extend(compute_recommendations(model, i, "", dictionary, reverse_dictionary, class_weights,model_rec, j, "", dictionary_rec, reverse_dictionary_rec, class_weights_rec))
        
    if (len(focused_elements)==1): 
        predictList=[]
        if direction=='right':
            previousPaths=getAllPaths(to_predict_parent_graph,focused_elements[0])
            for path in previousPaths:
                pub_t=compute_recommendations_OneDirection(model, path, "", dictionary, reverse_dictionary, class_weights)    
                for i in pub_t:
                    predictions.append(({'node':[i],
                        'connect':{i:[path[-1]]}},pub_t[i]))
        else:
            succPaths=getAllPaths(childGraph,focused_elements[0])
            for path in succPaths:
                pub_t=compute_recommendations_OneDirection(model_rec, path, "", dictionary_rec, reverse_dictionary_rec, class_weights_rec)    
                for i in pub_t:
                    predictions.append(({'node':[i],
                        'connect':{path[-1]:[i]}},pub_t[i]))
    predictions=list(sorted(predictions, key=lambda kv: kv[1], reverse=True))
    indexList=[]
    for i in predictions:
        if i[0]['node'] in indexList:
            predictions.remove(i)
        else:
            indexList.append(i[0]['node'])
    return predictions


class PredictionRequest(BaseModel):
    to_predict_parent_graph:Dict[str,List[str]]
    focused_elements:List[str]
    direction:str = "right"

@app.post("/PredictionRequest/")
def read_item(predictionRequest: PredictionRequest):
    print(newAlgo(predictionRequest.to_predict_parent_graph,predictionRequest.focused_elements,predictionRequest.direction))
    return newAlgo(predictionRequest.to_predict_parent_graph,predictionRequest.focused_elements,predictionRequest.direction)