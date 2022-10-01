# %%
import re
import csv
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import layers, losses, Model, callbacks
from transformers import BertTokenizer, TFBertModel, AdamWeightDecay

# %%
with open('./metadata.json', 'r') as f:
    metadata = json.load(f)

with open('./train.json', 'r') as f:
    file = json.load(f)
    
rep = {'—': '', '…': '', '“': ''} # define desired replacements here
# use these three lines to do the replacement
rep = dict((re.escape(k), v) for k, v in rep.items())
pattern = re.compile("|".join(rep.keys()))
preprocess = lambda text: pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

data = []
for count, dictionary in file.items():
    group = [list(d.values())[0] for d in dictionary]
    for i, d in enumerate(group):
        group[i]['utterance'] = preprocess(group[i]['utterance'])
        group[i]['prev_utterance'] = group[i-1]['utterance'] if i > 0 else ''
        group[i]['next_utterance'] = preprocess(group[i+1]['utterance']) if i < len(group)-1 else ''
    data += group
data = pd.DataFrame(data)

# encoder
relation_encoder, speaker_encoder, emotion_encoder = preprocessing.LabelEncoder(), preprocessing.LabelEncoder(), preprocessing.LabelEncoder()
onehot = preprocessing.OneHotEncoder()

# encode catgorical data
emotion_encoder.fit(metadata['emotion'])
relation_encoder.fit(metadata['relation'])
speaker_encoder.fit(data['speaker'])
all_speaker = speaker_encoder.transform(data['speaker'])
onehot.fit(all_speaker.reshape((-1, 1)))
        
# length specification
len_relation = len(metadata['relation'])
len_speaker = len(speaker_encoder.classes_)
len_ids = 64

def getAllCharacters(listeners):
    encoded = [0] * len_relation
    characters = relation_encoder.transform([d['relation'] for d in listeners])
    for c in characters:
        encoded[c] = 1
    return encoded

# tokenize
tokenizer_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

def tokenization(data, result):
    for u in ['prev_utterance', 'utterance', 'next_utterance']:
        x = tokenizer(data[u].tolist(), truncation=True, max_length=len_ids, padding="max_length", return_tensors="tf")
        result.update({f'{u}_input_ids': x['input_ids'], f'{u}_attention_mask': x['attention_mask']})
    return result

def createDataset(input):
    speaker = speaker_encoder.transform(input['speaker'])
    speaker = tf.convert_to_tensor(onehot.transform(speaker.reshape((-1, 1))).toarray())
    listener = tf.convert_to_tensor(input['listener'].apply(getAllCharacters).values.tolist())
    X = tokenization(input, {'speaker': speaker, 'listener': listener})
    Y = tf.convert_to_tensor(emotion_encoder.transform(input['emotion']))
    len_X, _ = X['utterance_input_ids'].shape
    dataset = Dataset.from_tensor_slices((X, Y)).shuffle(len_X).batch(8)
    return dataset

def kFoldData(train_index, valid_index):
    train, valid = data.iloc[train_index], data.iloc[valid_index]
    train_dataset = createDataset(train)
    valid_dataset = createDataset(valid)
    return train_dataset, valid_dataset

# %%
model_name = 'bert-base-chinese'
def modelConstruct(len_speaker, len_listeners, len_ids):
    speaker = layers.Input(name='speaker', shape=(len_speaker,))
    listener = layers.Input(name='listener', shape=(len_listeners,))
    prev_utterance_input_ids = layers.Input(name='prev_utterance_input_ids', shape=(len_ids,), dtype='int32')
    prev_utterance_attention_mask = layers.Input(name='prev_utterance_attention_mask', shape=(len_ids,), dtype='int32')
    utterance_input_ids = layers.Input(name='utterance_input_ids', shape=(len_ids,), dtype='int32')
    utterance_attention_mask = layers.Input(name='utterance_attention_mask', shape=(len_ids,), dtype='int32')
    next_utterance_input_ids = layers.Input(name='next_utterance_input_ids', shape=(len_ids,), dtype='int32')
    next_utterance_attention_mask = layers.Input(name='next_utterance_attention_mask', shape=(len_ids,), dtype='int32')
    bert = TFBertModel.from_pretrained(model_name)
    bert.trainable = False

    prev_utterance_embedding = bert(prev_utterance_input_ids, attention_mask=prev_utterance_attention_mask)[0]
    prev_utterance_embedding = prev_utterance_embedding[:, 0, :]
    utterance_embedding = bert(utterance_input_ids, attention_mask=utterance_attention_mask)[0]
    utterance_embedding = utterance_embedding[:, 0, :]
    next_utterance_embedding = bert(next_utterance_input_ids, attention_mask=next_utterance_attention_mask)[0]
    next_utterance_embedding = next_utterance_embedding[:, 0, :]
    embedding = layers.LSTM(800)(tf.stack([prev_utterance_embedding, utterance_embedding, next_utterance_embedding], axis=1))

    people = layers.Concatenate()([speaker, listener])
    people = layers.Dense(50, activation='relu')(people)
    people = layers.Dropout(0.15)(people)
    people = layers.Dense(50, activation='relu')(people)
    people = layers.Dropout(0.15)(people)

    embedding = layers.Concatenate()([embedding, people])
    embedding = layers.Dropout(0.15)(embedding)
    embedding = layers.Dense(400, activation='relu')(embedding)
    embedding = layers.Dropout(0.15)(embedding)
    embedding = layers.Dense(100, activation='relu')(embedding)
    output = layers.Dense(7, activation='softmax')(embedding)

    model = Model(inputs=[prev_utterance_input_ids, prev_utterance_attention_mask,
                          utterance_input_ids, utterance_attention_mask,
                          next_utterance_input_ids, next_utterance_attention_mask,
                          speaker, listener], outputs = output)
    model.compile(optimizer = AdamWeightDecay(learning_rate=1e-4), loss = losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
    return model

tf.debugging.set_log_device_placement(True)
models = []
def trainBert(num_model):
    kf = KFold(n_splits=num_model, shuffle=True, random_state=777)
    for i, (train_index, valid_index) in enumerate(kf.split(data)):
        print(i+1)
        print('preparing dataset')
        train_dataset, valid_dataset = kFoldData(train_index, valid_index)
        checkpoint_path = f'./model{i+1}.h5'
        model_checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True)
        save = [model_checkpoint_callback]
        sentiment_model = modelConstruct(len_speaker, len_relation, len_ids)
        history = sentiment_model.fit(train_dataset, epochs=10, validation_data=valid_dataset, callbacks=save)
        models.append(sentiment_model)

trainBert(3)

# %%
with open('./test.json', 'r') as f:
    file = json.load(f)
    
print('preprocessing test data')
test, id = [], []
for count, dictionary in file.items():
    id += [list(d.keys())[0] for d in dictionary]
    group = [list(d.values())[0] for d in dictionary]
    for i, d in enumerate(group):
        group[i]['utterance'] = preprocess(group[i]['utterance'])
        group[i]['prev_utterance'] = group[i-1]['utterance'] if i > 0 else ''
        group[i]['next_utterance'] = preprocess(group[i+1]['utterance']) if i < len(group)-1 else ''
    test += group
test = pd.DataFrame(test)
test['id'] = id

speaker = speaker_encoder.transform(test['speaker'])
speaker = tf.convert_to_tensor(onehot.transform(speaker.reshape((-1, 1))).toarray())
listener = tf.convert_to_tensor(test['listener'].apply(getAllCharacters).values.tolist())
X = tokenization(test, {'speaker': speaker, 'listener': listener})
X = Dataset.from_tensor_slices(X).batch(64)

voting = []
for i in range(3):
    print(f'model{i+1} prediction')
    prediction = []
    sentiment_model = modelConstruct(len_speaker, len_relation, len_ids)
    sentiment_model.load_weights(f'./model{i+1}.h5')
    for x in tqdm(X):
        prediction += list(np.argmax(sentiment_model.predict(x), axis=1))
    voting.append(prediction)

voting = np.array(voting)
def mostFreq(arr):
    unique, counts = np.unique(arr, return_counts=True)
    max_label, max_count = 0, 0
    for label, count in zip(unique, counts):
        if count > max_count:
            max_label, max_count = label, count
    return max_label
result = np.apply_along_axis(mostFreq, 0, voting)
result = emotion_encoder.inverse_transform(result)

with open('./result.csv', 'w', newline='') as f:
    writer = csv.writer(f)    
    writer.writerow(['id', 'emotion'])
    print('writing')
    for i, j in zip(id, result):
        writer.writerow([i, j])


