# importing necessary libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
# importing required models
from next_word_prediction_and_sentence_completion_new import *
from App_BERT_sentence_Corrention_model import *
# from sentence_correction_function import *

# initializing flask
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

# after pressing "correct" button
@app.route('/correct', methods=['POST'])
# function to correct the user input
def correct():
    # request to html 
    rf=request.form
    
    for key in rf.keys():
      data=key

    data=str(data)
    
    # data processing before sending it to our semtemce_correction function 
    data = data.replace("\"","")
    data = data.replace("n","")
    
    # invoking the "sentence_correction" function
    y=sentence_correction(data)
    y=str(y)

    # loading output data in json format
    data_dic=json.dumps(y, ensure_ascii=False)


    # processing the outputs
    data_dic = data_dic.replace("\"","")
    data_dic = data_dic.replace("\\", "")  

    # assigning the output in dictionary
    resp_dic={'data':data_dic}

    # returning output data in json format
    resp = jsonify(resp_dic)
    resp.headers['Access-Control-Allow-Origin']='*'
    return resp


# Next word and sentence prediction after pressing space bar
@app.route('/predict', methods=['POST'])
# function to predict the suggestions
def predict():
    
    # 
    rf1=request.form
   
    for key in rf1.keys():
      data1=key
    
    r1 = str(data1)
    pos1 = r1.rfind("।")
    pos2 = r1.rfind("!")
    pos3 = r1.rfind("?")
    max_pos = max(pos1,pos2,pos3)
    last_line = r1[max_pos + 1 : len(r1)]
    print(last_line)

    # invoking the model functions
    y1=nxt_word_prediction(last_line)
    y2=sentence_completion(last_line)
    
    # loading outputs in json format
    data_dic1=json.dumps(y1, ensure_ascii=False)
    data_dic2=json.dumps(y2, ensure_ascii=False)
    
    # processing the outputs
    data_dic2 = data_dic2.replace("\"","")
    data_dic2 = data_dic2.replace("\\", "")   

    resp_dic1={'data':data_dic1, 'data2':data_dic2}
    
    # returning output data in json format
    resp1 = jsonify(resp_dic1)
    resp1.headers['Access-Control-Allow-Origin']='*'
    return resp1


# executing script
if __name__ == "__main__":
    app.run(debug=True)