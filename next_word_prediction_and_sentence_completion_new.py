import pickle
import numpy as np
from random import random
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import math
import tensorflow as tf

dic="word_prediction_pickles/"

with open(dic+'dict_Of_index_Top_Words.pickle', 'rb') as handle:
    dict_Of_index_Top_Words = pickle.load(handle)
with open(dic+'top_k_word.pickle', 'rb') as handle:
    top_k_word = pickle.load(handle)
with open(dic+'tokenizer.pickle', 'rb') as handle:
    Tokenizer = pickle.load(handle)

def PPL(y_true, y_pred):
    return K.pow(2.0, K.mean(K.categorical_crossentropy(y_true, y_pred,from_logits=True )))


model = keras.models.load_model('Models/Next Word prediction Full CNN model',
                                              custom_objects={'PPL':PPL})

def padding_word_w(dem,length): 
  dem=dem.strip()
  word=[]
  word.append(Tokenizer.word_index.get('#'))
  for y in dem:
    value_check=Tokenizer.word_index.get(y)
    if(value_check==None):
      value_check=len(Tokenizer.word_index)+1
    word.append(value_check)
  word.append(Tokenizer.word_index.get('$'))

  flag=0
  out=[]
  if (len(word)<12):
    word_len=len(word)
    dif=12-word_len
    flag=1
    for c in range(0,math.ceil(dif/2)):
      out.append(0)
    for c in word:
      out.append(c)
    for c in range(0,math.floor(dif/2)):
      out.append(0)

         
  if (len(word)>=12 and flag==0):   
    out=word[0:12]
    out[11]=Tokenizer.word_index.get('$')
  return out

def padding_sentance_w(dem,length):
  x=len(dem)
  if(x<14):    
    pad_sen=[]
    dif=14-x
    for i in range(0,dif):
      pad_sen.append([0,0,0,0,0,0,0,0,0,0,0,0])
    for c in dem:
      pad_sen.append(c)
  else:
    pad_sen=dem[0:14]
  return pad_sen

def mask_sen(pad_sen):
  pad_zero=[]
  for i in range (0,256):
    pad_zero.append(float(0.0))

  pad_one=[]
  for i in range (0,256):
    pad_one.append(float(1.0))

  full_sen=[]
  for i in pad_sen:
    all_zero=0
    for j in i:
       if(j==0):
         all_zero=all_zero+1
    if (all_zero==len(i)):
      full_sen.append(pad_zero)
    else:
      full_sen.append(pad_one)
  
  return full_sen

def x_y_genarator_model_2(one_sentence):
    input1=[]
    for j in one_sentence:
      input1.append(padding_word_w(j,10))
    input11=padding_sentance_w(input1,10)
    mask=mask_sen(input11)
      
    return input11,mask

def nxt_word_prediction(sen):
  final_sen=""
  #sen=sen.strip()
  for i in sen:
    if i in ["।","?","!"] :
      final_sen=""
    else:
      final_sen=final_sen+i
  sen_words=final_sen.split()
  
  if len(sen_words)>1:
    a,b=x_y_genarator_model_2(sen_words)
    a=np.array(a)
    b=np.array(b)
    a=np.reshape(a, (1, 14, 12))
    b=np.reshape(b, (1,14,256))

    Y = model.predict([[a],b])
    Y=np.reshape(Y, (len(top_k_word)))
    ind = np.argpartition(Y, -3)[-3:]
    res=[]
    for i in ind:
      res.append (dict_Of_index_Top_Words[i])
    return res
  else:
    return ""


# input_sen="আজ রোববার সকাল থেকে শুরু"
# input="প্রতিবেদন প্রকাশ করেছে। ইংরেজি অক্ষর ক্রম অনুযায়ী। প্রতিবেদন প্রকাশ করেছে"

# predic=nxt_word_prediction(input_sen)

# print(predic)

def Initialize_prob_list(tmp,beam_width):
  prob_list=[]
  for i in range(0,beam_width):
    prob_list.append([1,tmp])
  return prob_list

def Append_to_Problist(prob_list,ind,probability):
  new_prob_list=[]
  for i in range(0,len(prob_list)):
    x=prob_list[i]
    temp=[]
    for j in x[1]:
      temp.append(j)
    temp.append(dict_Of_index_Top_Words[ind[i]])
    new_prob_list.append([probability[i]*x[0],temp])
  return new_prob_list


def predic_word(sen):
  a,b=x_y_genarator_model_2(sen)
  a=np.array(a)
  b=np.array(b)
  a=np.reshape(a, (1, 14, 12))
  b=np.reshape(b, (1,14,256))
  Y = model.predict([[a],b])

  yy=np.reshape(Y, len(top_k_word))

  return Y,yy

beam_width = 5
def check_end_char(prob):
  sen=[]
  for j in range(0,3):
    i=prob[j] 
    if i[1][-1] in ["।","?","!"] :
      if len(sen)==0:
        sen=i
      else:
        if sen[0]<i[0]:
          sen=i
  if len(sen)==0:
    return None
  return sen[1]

def sentence_completion(sen):
  final_sen=""
  #sen=sen.strip()
  for i in sen:
    if i in ["।","?","!"] :
      final_sen=""
    else:
      final_sen=final_sen+i
  
  sen_words=final_sen.split()
  
  if len(sen_words)>1:
    prob_list=Initialize_prob_list(sen_words,beam_width)
    
    Y,yy=predic_word(sen_words)
    
    ind = np.argpartition(yy, -beam_width)[-beam_width:]
    probability=[]
    for j in ind:
      probability.append(Y[0][j])
    prob_list=Append_to_Problist(prob_list,ind,probability)
    
    check=check_end_char(prob_list)
    
    if check!=None:
      output=""
      
      for k in check:
        if k in ["।","?","!"] :
          output=output+""+k
          break
        else:
          output=output+" "+k
      return output.strip()

    for i in range(0,4):
      temp_prob_list = []
      index_up=0
      for tuples in prob_list:
      
        Y,yy=predic_word(tuples[1])
        
        ind = np.argpartition(yy, -beam_width)[-beam_width:]
        probability=[]
        for j in ind:
          probability.append(Y[0][j])
        ini_prod=Initialize_prob_list(tuples[1],beam_width)
        new_prob=Append_to_Problist(ini_prod,ind,probability)
        for l in new_prob:   
          temp_prob_list.append(l)  

      index_up=index_up+1
      demo=sorted(temp_prob_list, key=lambda x: x[0])
      ch= demo[::-1]
      
      prob_list=ch[0:beam_width]
      check=check_end_char(ch)
      if check!=None:
        output=""
        for k in check:
          if k in ["।","?","!"] :
            output=output+""+k
            break
          else:
            output=output+" "+k

        return output.strip()

    best=prob_list[0][1]
    best_prob=prob_list[0][0]

    for i in prob_list:
      if i[0]> best_prob:
        best_prob=i[0]
        best=i[1]
    output=""
    for i in best:    
      if i in ["।","?","!"] :
        output=output+""+i
        break
      else:
        output=output+" "+i
    
    return output.strip()  

  else:
    return None

# input_1="এক সংবাদ বিজ্ঞপ্তিতে "
# cc=sentence_completion(input_1)
# print(cc)