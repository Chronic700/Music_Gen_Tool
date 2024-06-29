import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
#assert len(tf.config.list_physical_devices('GPU'))>0

start=time.time()
songs=mdl.lab1.load_training_data()
print("Example song:")
print(songs[0])
#mdl.lab1.play_song(songs[0])
songs_joined = "\n\n".join(songs) 

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")
print(songs[1])
char2idx={u:i for i,u in enumerate(vocab)}
idx2char=np.array(vocab)
print('{')
for char,_ in zip(char2idx, range(83)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n')
def vectorize_string(string):
    vector=np.array([char2idx[char] for char in string])
    return vector

def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

    
    LSTM(rnn_units),

    tf.keras.layers.Dense(vocab_size)
    ])

  return model

def get_batch(vectorized_songs, seq_length, batch_size):
    n=vectorized_songs.shape[0]-1
    idx=np.random.choice(n-seq_length,batch_size)
    input_batch=[vectorized_songs[i:i+seq_length] for i in idx]
    output_batch=[vectorized_songs[i+1:i+seq_length+1] for i in idx]
    
    x_batch=np.reshape(input_batch,[batch_size, seq_length])
    y_batch=np.reshape(output_batch,[batch_size, seq_length])
    return x_batch, y_batch

vocab_size=len(vocab)
embedding_dim= 256
rnn_units=1024

model=build_model(vocab_size,embedding_dim,rnn_units,batch_size=1)
model.summary()

# model= tf.keras.models.load_model("C:\\Users\\Shikhar Gupta\\Desktop\\Tensorflow\\Model_200B.h5")

def generate_text(model, start_string, generation_length=1000):
    input_eval = [char2idx[s] for s in start_string] 
    
    input_eval = tf.expand_dims(input_eval, 0)

    
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
      
        text_generated.append(idx2char [predicted_id]) 
        
            
    return (start_string + ''.join(text_generated))

model.summary()

generated_text = generate_text(model, start_string="X:", generation_length=1000)

generated_songs=mdl.lab1.extract_song_snippet(generated_text)

print(generated_text)


for i, song in enumerate(generated_songs):
    waveform=mdl.lab1.play_song(song)
    if waveform:
        print("Generated song", i)
        ipythondisplay.display(waveform)