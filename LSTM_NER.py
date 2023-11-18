import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from nltk.corpus import conll2002

# Download the NER dataset from NLTK
nltk.download('conll2002')

# Load the dataset
data = conll2002.iob_sents()

# Separate words, POS tags, and NER labels
X = [[(word, pos) for word, pos, ner in sent] for sent in data]
y = [[ner for word, pos, ner in sent] for sent in data]

# Define vocabulary and label sets
words = set([word for sent in X for word, pos in sent])
labels = set([ner for sent_labels in y for ner in sent_labels])

# Create word and label dictionaries
word2idx = {word: idx + 1 for idx, word in enumerate(words)}
label2idx = {label: idx for idx, label in enumerate(labels)}

# Convert words and labels to indices
X_idx = [[word2idx[word] for word, pos in sent] for sent in X]
y_idx = [[label2idx[ner] for ner in sent_labels] for sent_labels in y]

# Pad sequences for uniform length
X_padded = pad_sequences(X_idx, padding='post')
y_padded = pad_sequences(y_idx, padding='post', value=label2idx['O'])

# Convert labels to one-hot encoding
y_one_hot = to_categorical(y_padded, num_classes=len(labels))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_one_hot, test_size=0.2, random_state=42)

# Build the LSTM-based NER model
model = Sequential()
model.add(Embedding(input_dim=len(words) + 1, output_dim=50, input_length=X_padded.shape[1]))
model.add(Dropout(0.1))
model.add(LSTM(100, return_sequences=True))
model.add(TimeDistributed(Dense(len(labels), activation='softmax')))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
