from  Dataset import x,y

import csv
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BasicTokenizer
from transformers import TFBertModel, TFBertPreTrainedModel, TFBertForSequenceClassification
from transformers import glue_convert_examples_to_features, InputExample
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import GlobalMaxPool1D
import os
import keras
import tensorflow as tf
from datetime import datetime
import json
from __future__ import unicode_literals

def load_data(file_name):
    """
        Read the CSV and creates a Panda dataframe fro the file content.
    """
    f = open(file_name, 'r')
    data = pd.read_csv(f, delimiter=',', encoding='utf-8')
    return data

df = load_data('/content/drive/My Drive/idehhub/Sentiment/Bert/data/taghche.csv')

# Remove the unnecessary columns
df = df.drop(columns=['date', 'bookname', 'bookID', 'like'])
# df = df.rename(columns={'Text':'text','Suggestion': 'label'})
df.loc[(df.rate < 2), 'label'] = '0'
df.loc[(df.rate < 4) & (df.rate>1), 'label'] = '1'
df.loc[(df.rate > 3), 'label'] = '2'
df = df.drop(columns=['rate'])
df.head()

df_pos = df.loc[df.label == '2']
df_neut = df.loc[df.label == '1']
df_neg = df.loc[df.label == '0']
print('Postive examples: {}  Negative examples: {} Neutral examples: {}' .format(len(df_pos),len(df_neg),len(df_neut)))

def create_small_dataset(df_pos, df_neg,df_neut, n_samples):
    """ Create a custom dataset of size `n_samples` from positive `df_pos` and negative `df-neg`
        examples.
    """
    duplicates = set()
    counter = 0
    data = {}
    data['comment'] = []
    data['polarity'] = []
    while counter < n_samples:
        index = np.random.randint(0, len(df_pos))
        if index in duplicates:
            continue
        row = df_pos.iloc[index]
        comment = remove_emoji(row['comment'])
        label = row['label']
        data['comment'].append(comment)
        data['polarity'].append(label)
        duplicates.add(index)
        counter += 1

    duplicates.clear()
    counter = 0
    while counter < n_samples:
        index = np.random.randint(0, len(df_neg))
        if index in duplicates:
            continue
        row = df_neg.iloc[index]
        comment = remove_emoji(row['comment'])
        label = row['label']
        data['comment'].append(comment)
        data['polarity'].append(label)
        duplicates.add(index)
        counter += 1
    duplicates.clear()
    counter = 0
    while counter < n_samples:
        index = np.random.randint(0, len(df_neut))
        if index in duplicates:
            continue
        row = df_neut.iloc[index]
        comment = remove_emoji(row['comment'])
        label = row['label']
        data['comment'].append(comment)
        data['polarity'].append(label)
        duplicates.add(index)
        counter += 1
    return pd.DataFrame.from_dict(data)

def remove_emoji(text):
    """ Remove a number of emojis from text."""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text = re.sub(emoji_pattern, ' ', text).replace('.','')
    return re.sub(r'[a-z]+[A-Z]+', '', text, re.I)

dataset_size = 10000
dataset = create_small_dataset(df_pos, df_neg, df_neut, dataset_size)

#shuffle the dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset

train_data, test_data = train_test_split(dataset, test_size=0.2)
print('train data size: {}    test data size: {}'.format(len(train_data), len(test_data)))

##digikala dataset
# define documents
with open('/content/drive/My Drive/idehhub/Sentiment/Sentiment-lstm/data/MirasOpinion/all_comments.txt') as f:
    tweets = f.read()
docs=tweets.splitlines()
# define class labels
with open('/content/drive/My Drive/idehhub/Sentiment/Sentiment-lstm/data/MirasOpinion/all_labels.txt') as f:
    labels = f.read()
labels=labels.splitlines()
labels=[int(label) for label in labels]

data=[]
for i in range(len(docs)) :
  data.append([docs[i],labels[i]])

#ballancing digi data
ballanced_data=[]
counter_pos=0
counter_neg=0
counter_neut=0
n_samples=14882
for i in range(len(data)):
  if(data[i][1]==1 and counter_pos < n_samples):
    ballanced_data.append(data[i])
    counter_pos+=1
  if(data[i][1]==-1 and counter_neg < n_samples):
    ballanced_data.append(data[i])
    counter_neg+=1
  if(data[i][1]==0 and counter_neut < n_samples):
    ballanced_data.append(data[i])
    counter_neut+=1

counter_pos

counter_neg

counter_neg

counter_neut

counter_pos

train_data, test_data = train_test_split(data, test_size=0.1)

print('train data size: {}    test data size: {}'.format(len(train_data), len(test_data)))

##end digi data

def convert_data_into_input_example(data):
    """ Covert the list of examples into a list of `InputExample` objects that is suitable
        for BERT model."""
    input_examples = []
    for i in range(len(data)):
        example = InputExample(
            guid= None,
            text_a= data.iloc[i]['comment'],
            text_b= None,
            label= data.iloc[i]['polarity']
        )
        input_examples.append(example)
    return input_examples

#digi
def convert_data_into_input_example_for_digi(data):
    """ Covert the list of examples into a list of `InputExample` objects that is suitable
        for BERT model."""
    input_examples = []
    for i in range(len(data)):
        example = InputExample(
            guid= None,
            text_a= data[i][0],
            text_b= None,
            label= f'{data[i][1]}'
        )
        input_examples.append(example)
    return input_examples

train_input_examples = convert_data_into_input_example(train_data)
val_input_examples = convert_data_into_input_example(test_data)

#digi
train_input_examples = convert_data_into_input_example_for_digi(train_data)
val_input_examples = convert_data_into_input_example_for_digi(test_data)

train_input_examples

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

text =  'گند بزنن به بانک سامان با اپ رمزینه و سیستم ارسال رمز پویا. یک بار نشد درست کار کنه لعنتی، اه 🤬🤬🤬🤬 #بانک_سامان'
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
text_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
print('text ids:', text_ids)
text_ids_with_special_tokens = tokenizer.build_inputs_with_special_tokens(text_ids)
print('text ids with special tokens: ', text_ids_with_special_tokens)

MAX_SEQ_LENGTH = 250
encoded_bert_text = tokenizer.encode(text, add_special_tokens=True, max_length=MAX_SEQ_LENGTH)
# encoded_bert_text = tokenizer.encode(text, add_special_tokens=True, max_length=MAX_SEQ_LENGTH, return_tensors='tf')

print('encoded text: ', encoded_bert_text)
decoded_text_with_special_token = tokenizer.decode(encoded_bert_text)
decoded_text_without_special_token = tokenizer.decode(encoded_bert_text, skip_special_tokens=True)

print('decoded text with special token: ', decoded_text_with_special_token)
print('decoded text without special token: ', decoded_text_without_special_token)

label_list = ['0','1','2']

bert_train_dataset = glue_convert_examples_to_features(examples=train_input_examples, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, task='mrpc', label_list=label_list)
bert_val_dataset = glue_convert_examples_to_features(examples=val_input_examples, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, task='mrpc', label_list=label_list)

for i in range(3):
#     print('Example: {}'.format(bert_train_dataset[i]))
    print('Example: {')
    print(' Input_ids: {}'.format(bert_train_dataset[i].input_ids))
    print(' attention_mask: {}'.format(bert_train_dataset[i].attention_mask))
    print(' token_type_ids: {}'.format(bert_train_dataset[i].token_type_ids))
    print(' label: {}'.format(bert_train_dataset[i].label))
    print('}')

#digi
label_list = ['-1','0','1']

bert_train_dataset = glue_convert_examples_to_features(examples=train_input_examples, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, task='mrpc', label_list=label_list)
bert_val_dataset = glue_convert_examples_to_features(examples=val_input_examples, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, task='mrpc', label_list=label_list)

for i in range(3):
#     print('Example: {}'.format(bert_train_dataset[i]))
    print('Example: {')
    print(' Input_ids: {}'.format(bert_train_dataset[i].input_ids))
    print(' attention_mask: {}'.format(bert_train_dataset[i].attention_mask))
    print(' token_type_ids: {}'.format(bert_train_dataset[i].token_type_ids))
    print(' label: {}'.format(bert_train_dataset[i].label))
    print('}')

ex = bert_train_dataset[0]
in_ids = ex.input_ids
decoded_sentence = tokenizer.decode(in_ids, skip_special_tokens=True)
print(decoded_sentence)

model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=3)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model.load_weights('/content/drive/My Drive/idehhub/Sentiment/Bert/SavedModel/BertSentiment2classModelNotBallancedEpoch1.h5')

def my_solution(bdset):
    """ Create a list of input tensors required to be in the first argument of the
        model call function for training. e.g. `model([input_ids, attention_mask, token_type_ids])`.
    """
    input_ids, attention_mask, token_type_ids, label = [], [], [], []
    for in_ex in bdset:
        input_ids.append(in_ex.input_ids)
        attention_mask.append(in_ex.attention_mask)
        token_type_ids.append(in_ex.token_type_ids)
        label.append(in_ex.label)

    input_ids = np.vstack(input_ids)
    attention_mask = np.vstack(attention_mask)
    token_type_ids = np.vstack(token_type_ids)
    label = np.vstack(label)
    return ([input_ids, attention_mask, token_type_ids], label)

def example_to_features(input_ids, attention_masks, token_type_ids, y):
    """ Convert a training example into the Bert compatible format."""
    return {"input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids},y

# Cell
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

EPOCHS = 2
for i in range(EPOCHS):
  x_train, y_train = my_solution(bert_train_dataset)
  x_val, y_val = my_solution(bert_val_dataset)

  print('x_train shape: {}'.format(x_train[0].shape))
  print('x_val shape: {}'.format(x_val[0].shape))

  train_ds = tf.data.Dataset.from_tensor_slices((x_train[0], x_train[1], x_train[2], y_train)).map(example_to_features).shuffle(100).batch(32)
  val_ds   = tf.data.Dataset.from_tensor_slices((x_val[0], x_val[1], x_val[2], y_val)).map(example_to_features).batch(64)

  print('Format of model input examples: {} '.format(train_ds.take(1)))


  model.fit(train_ds, validation_data=val_ds, epochs=1)
  model.save_weights(f'/content/drive/My Drive/idehhub/Sentiment/Bert/SavedModel/BertSentiment2classModelNotBallancedEpoch250MaxLength{i}.h5')

def example_to_features_predict(input_ids, attention_masks, token_type_ids):
    """
        Convert the test examples into Bert compatible format.
    """
    return {"input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids}

from scipy.special import softmax
def get_prediction(in_sentences):
    """
        Prepare the test comments and return the predictions.
    """
    labels = ["-1","0", "1"]
    input_examples = [InputExample(guid="", text_a = x, text_b = None, label = '0') for x in in_sentences] # here, "" is just a dummy label
    predict_input_fn = glue_convert_examples_to_features(examples=input_examples, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, task='mrpc', label_list=label_list)
    x_test_input, y_test_input = my_solution(predict_input_fn)
    test_ds   = tf.data.Dataset.from_tensor_slices((x_test_input[0], x_test_input[1], x_test_input[2])).map(example_to_features_predict).batch(32)

    predictions = softmax((model.predict(test_ds))[0])
    #   print('predictions:', predictions[0].shape)
    # predictions_classes = np.argmax(predictions[0], axis = 1)
    # return [(sentence, prediction) for sentence, prediction in zip(in_sentences, predictions_classes)]
    return predictions
