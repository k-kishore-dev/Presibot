import json
import pickle
import random

import numpy as np
import nltk
from nltk.stem import LancasterStemmer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

nltk.download("punkt")
stemmer = LancasterStemmer()

# Load intents from JSON
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("chatbot.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)
except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)
    training = []
    output = []
    output_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("chatbot.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)

# Load or create the model
try:
    myChatModel = Sequential()
    myChatModel.add(Dense(8, input_shape=[len(words)], activation="relu"))
    myChatModel.add(Dense(len(labels), activation="softmax"))
    myChatModel.load_weights("chatbotmodel.weights.h5")
    print("Loaded model weights from disk")
except (FileNotFoundError, ValueError):
    myChatModel = Sequential()
    myChatModel.add(Dense(8, input_shape=[len(words)], activation="relu"))
    myChatModel.add(Dense(len(labels), activation="softmax"))

    myChatModel.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    myChatModel.fit(training, output, epochs=1000, batch_size=8)
    myChatModel.save_weights("chatbotmodel.weights.h5")
    print("Saved model weights to disk")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    current_text = bag_of_words(message, words)
    current_text_array = [current_text]
    numpy_current_text = np.array(current_text_array)

    if np.all((numpy_current_text == 0)):
        return jsonify({'response': "I didn't get that."})

    result = myChatModel.predict(numpy_current_text[0:1])
    result_index = np.argmax(result)
    tag = labels[result_index]

    if result[0][result_index] > 0.7:
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        return jsonify({'response': random.choice(responses)})
    else:
        return jsonify({'response': "I didn't get that."})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
