# nn-generative
A trained neural network generative text. Note: it's still not as accurate as other AI does, so this is for educational purposes only. With Tensorflow, numpy, pandas, and nltk. To use the trained model, do the following: ```python
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import random
model = load_model("AI_MODEL.keras")```
