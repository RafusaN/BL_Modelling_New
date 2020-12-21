import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, Conv1D, GlobalMaxPooling1D, Multiply, BatchNormalization, Bidirectional, GRU, Add, Multiply, Subtract, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.activations import softmax
import numpy as np
from tensorflow.keras.utils import plot_model
import pickle

from gensim.models.fasttext import FastText
from random import random, choice

dic="correction_pickles/"


with open(dic+'dict_Of_index_Top_Words.pickle', 'rb') as handle:
    dict_Of_index_Top_Words_1 = pickle.load(handle)


root="Models/Fast-Text-Saved-Model/"

fastText_model=FastText.load(root+"/new_ft_model_2")

model_path = 'Models/SC-BERT-FINE-TUNE-test-model'
pretrained_BERT = keras.models.load_model(model_path)



def padding_sentance(dem):
    c=0
    if (len(dem)<15):  
        x=len(dem)
        pad_sen=[]
        
        dif=15-x
        for i in range(0,dif):
            c=c+1
            pad_sen.append([float(0.0)]*512) 
        for i in dem:
            pad_sen.append(i)

    else:
        pad_sen=dem[0:15] 
        #print("from padd if ",len(pad_sen))
    pad_sen=np.array(pad_sen)
    #print(error)
    return pad_sen,c

def x_y_genarator_model(one_sentence):
    input=[]
    words=one_sentence.split()
    #print(words)
    for i in range ( len(words) ):    
        x=words[i] 
        for k in x:
            if(k=="০" or k=="১" or k=="২" or k=="৩" or k=="৪" or k=="৫" or k=="৬" or k=="৭" or k=="৮" or k=="৯"):
                x="1111111111"
                break
        #print(x)  
        try:
            input.append(fastText_model.wv[x])
        except:        
            input.append([0.5]*512)
        
    input,c=padding_sentance(input)

    return input

      

def sentence_correction(str):
  inputs=[]
  sentence=""
  output=""
  for i in str:
    if i =="।" or i=="!" or i=="?":
      sentence=sentence+" "+i
      inputs.append(sentence)
      sentence=""
    else:
      sentence=sentence+i

  for j in inputs:
    sen_words=j.split()
    x=x_y_genarator_model(j)
    c=[]
    c.append(x)
    x=np.array(c)
    yy=pretrained_BERT.predict([x])
    check=yy[1]
    check=np.array(check)
    result=[]
    for k in check[0]:
      max=-1
      index=-1
      o=np.array(k)
      for j in range(40002):
        if o[j]>max:
          index=j
          max=o[j]
      result.append(index)
    index_count=0
    for k in result:
      p=dict_Of_index_Top_Words_1[k]
      if p!="PAD" and p!="UNK" and p!="1111111111":
        index_count=index_count+1
        if p=="।" or p=="?" or p=="!":
          output=output+p
        else:
          output=output+" "+p
      if p=="UNK" or p=="1111111111":
        output=output+" "+sen_words[index_count]
        index_count=index_count+1
        
  return  output.strip()

  #sentence_correction("গত বছরের ১১১১১ শুরুতেও --- হাতে প্রায় সয়া লাখ কতি তাকা অতিরিক্ত তারল্ল ছিল। বিনিয়গ করতে না পারায় ব্যাঙ্কগুল আমানতের সুধার কমিয়ে দেয়।")
