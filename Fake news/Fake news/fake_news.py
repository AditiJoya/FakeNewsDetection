import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from nltk.corpus import wordnet

np.random.seed(3)

df= pd.read_csv('main_data1.tsv', delimiter='\t',encoding='utf-8')
df=pd.read_excel('FN2.xlsx')


#df.to_csv(path_or_buf='main_data.tsv',sep='\t',encoding='utf-8')

"""con=df["Class"].isna() 
rev_con=np.invert(con)
df=df[rev_con]"""     






df = df.sample(frac=1, random_state=42)
df=df.head(50) #replace the value with 400

Y = df.iloc[:, 6:7].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)





doc_complete=df['Description ']
doc_clean = [doc.split() for doc in doc_complete]


list1=[]
for doc in doc_clean:
    list1=list1+doc
words = list(set(list1))


#words.append("ENDPAD")
word2idx = {w: i for i, w in enumerate(words)}





X=[]
for doc in doc_clean:
    temp=[]
    for j in doc:
        temp.append(word2idx[j])
        
    X.append(temp)
    
    
    
maxlen = max([len(s) for s in X])

# Need not to convert it to Np.array
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(sequences=X, padding="post",value=1316)

Y=np.array(Y)

vocab_size=len(words)+1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)


from keras.models import Sequential,Input,Model
from keras.layers import Dense
from keras.layers import Flatten,Dropout,LSTM,Activation,GRU,SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
               



#create RNN
model = Sequential()

#model.add(Dense(32, input_dim=320))
model.add(Embedding(vocab_size,32,input_length=maxlen))#repal
#model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=320, trainable=True))
model.add(GRU(300))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Fit the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#out=model.predict_classes(X_test)

y_prob = model.predict(X_test) 

y_pred = np.argmax(y_prob, axis=1)




from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')


#visualize the confusion matrix
import seaborn as sn
import pandas  as pd
 
 
df_cm = pd.DataFrame(cm, range(2),
                  range(2))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 4})# font size
plt.show()

