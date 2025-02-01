from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import random

textDF = pd.read_csv("dataset.csv")
text = list(textDF.text.values)
joinedText = " ".join(text)

data_train = joinedText[:100000]
tokenizer = RegexpTokenizer(r"\w+")
token = tokenizer.tokenize(data_train.lower())
uniqueTokens = np.unique(token)
uniqueTokensIndex = {token: idx for idx, token in enumerate(uniqueTokens)}

inputWords = []
nextWords = []
bestPredictedWords = 10

for i in range(len(token) - bestPredictedWords):
    inputWords.append(token[i:i + bestPredictedWords])
    nextWords.append(token[i + bestPredictedWords])

X_train = np.zeros((len(inputWords), bestPredictedWords, len(uniqueTokens)), dtype=bool)
y_train = np.zeros((len(nextWords), len(uniqueTokens)), dtype=bool)

for i, words in enumerate(inputWords):
    for j, word in enumerate(words):
        X_train[i, j, uniqueTokensIndex[word]] = 1
    y_train[i, uniqueTokensIndex[nextWords[i]]] = 1

try:
    model = load_model("AI_MODEL.keras")
except Exception:
    model = Sequential()
    model.add(LSTM(128, input_shape = (bestPredictedWords, len(uniqueTokens)), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(units=len(uniqueTokens), activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
    checkpoint = ModelCheckpoint("AI_MODEL.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose = 1)

    history = model.fit(X_train, y_train, batch_size = 128, epochs=10, shuffle=True, validation_split=0.2, callbacks=[checkpoint])

def predictNextWord(inputText):
    inputText = inputText.lower()
    inputTokens = tokenizer.tokenize(inputText)

    if len(inputTokens) < bestPredictedWords:
        inputTokens = ([''] * (bestPredictedWords - len(inputTokens))) + inputTokens
    
    X = np.zeros((1, bestPredictedWords, len(uniqueTokens)), dtype=bool)

    for i, word in enumerate(inputTokens[-bestPredictedWords:]):
        if word in uniqueTokensIndex:
            X[0, i, uniqueTokensIndex[word]] = 1
    
    predictions = model.predict(X)[0]

    temperature = 1.0
    predictions = np.log(predictions + 1e-7) / temperature
    predictions = np.exp(predictions) / np.sum(np.exp(predictions))
    
    return np.random.choice(len(predictions), p=predictions)

def generateText(inputText, textLength, creativity=5):
    wordSequence = inputText.split()
    current = 0
    for _ in range(textLength):
        subSequence = " ".join(tokenizer.tokenize(" ".join(wordSequence).lower())[current:current + bestPredictedWords])
        try:
            choice = uniqueTokens[predictNextWord(subSequence, creativity)]
        except:
            choice = random.choice(uniqueTokens)
        wordSequence.append(choice)
        current += 1
    return " ".join(wordSequence)

userInput = input("Enter a sentence for the AI to continue: ")
response = generateText(userInput, 50, 5)
print("AI:", response)