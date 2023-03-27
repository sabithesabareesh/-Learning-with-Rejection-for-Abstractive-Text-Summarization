import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Load data
data = open("text.txt", "r").read()
summaries = open("summary.txt", "r").read()

# Tokenize input and summary data
max_len_text = 500
max_len_summary = 50

tokenizer_text = Tokenizer()
tokenizer_text.fit_on_texts(data)
input_seq = tokenizer_text.texts_to_sequences(data)
input_seq = pad_sequences(input_seq, maxlen=max_len_text, padding='post')

tokenizer_summary = Tokenizer()
tokenizer_summary.fit_on_texts(summaries)
target_seq = tokenizer_summary.texts_to_sequences(summaries)
target_seq = pad_sequences(target_seq, maxlen=max_len_summary, padding='post')

# Define the model
latent_dim = 256

encoder_inputs = Input(shape=(max_len_text,))
encoder_masking = Masking(mask_value=0.0)(encoder_inputs)
encoder_embedding = Embedding(input_dim=len(tokenizer_text.word_index)+1, output_dim=latent_dim, mask_zero=True)(encoder_masking)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_masking = Masking(mask_value=0.0)(decoder_inputs)
decoder_embedding = Embedding(input_dim=len(tokenizer_summary.word_index)+1, output_dim=latent_dim, mask_zero=True)(decoder_masking)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(tokenizer_summary.word_index)+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Define the loss function and compile the model
def masked_cross_entropy_loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    mask = tf.cast(mask, dtype=tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    loss = tf.multiply(loss, mask)
    loss = tf.divide(tf.reduce_sum(loss), tf.reduce_sum(mask))
    return loss

model.compile(optimizer='adam', loss=masked_cross_entropy_loss)

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit([input_seq, target_seq[:,:-1]], target_seq[:,1:], epochs=10, callbacks=[early_stopping], validation_split=0.2)

# Apply "Learning with Rejection"
# Define a function to calculate the model's confidence on each example
def confidence_scores(model, input_seq):
    prediction = model.predict([input_seq, np.zeros((len(input_seq), max_len_summary-1))])
    conf_scores = []
    for i in range(len(prediction)):
        conf_score = 1 - prediction[i][np.arange(len(prediction[i])), np.argmax(prediction[i], axis=1)].sum() / prediction.shape[1]
        conf_scores.append(conf_score)
    return conf_scores

# Define a function to reject examples with low confidence scores
def reject_samples(model, input_seq, target_seq, threshold):
  conf_scores = confidence_scores(model, input_seq)
  indices_to_keep = np.where(np.array(conf_scores) >= threshold)[0]
  input_seq_filtered = input_seq[indices_to_keep]
  target_seq_filtered = target_seq[indices_to_keep]
  return input_seq_filtered, target_seq_filtered
  
#Apply "Learning with Rejection" to filter out low-confidence examples
threshold = 0.8
input_seq_filtered, target_seq_filtered = reject_samples(model, input_seq, target_seq, threshold)

#Train the model again on the filtered data
model.fit([input_seq_filtered, target_seq_filtered[:,:-1]], target_seq_filtered[:,1:], epochs=10, callbacks=[early_stopping], validation_split=0.2)
