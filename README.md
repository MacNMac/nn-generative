# nn-generative-python
A trained neural network generative text. Note: it's still not as accurate as other AI does, so this is for educational purposes only. With Tensorflow, numpy, pandas, and nltk. <b>To use the trained model, do the following:</b><br>
### Installation:
```bash
pip install tensorflow pandas nltk
```

```python
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import random
```
### Load the model:
```python
model = load_model("AI_MODEL.keras")
```

If you want to train a total different model, you can delete the 'AI_MODEL.keras' and run the 'generative.py' code. It'll automatically trains a new one depending your likings.
### How to modify:
You can change these settings such as 'bestPredictedWords', 'batch_size' or the 'LSTM' value
```python
bestPredictedWords = 10
```
```python
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
```
