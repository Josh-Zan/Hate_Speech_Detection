#import libraries
from unittest import result
from flask import *
import pandas as pd
from keras_preprocessing.text import Tokenizer
from keras.models import model_from_json
from keras_preprocessing.sequence import pad_sequences

app = Flask(__name__)  # creating the Flask class object

data = pd.read_csv('MCCNN/data.csv')
train_size = int(data.shape[0] * 0.8)
train_df = data[:train_size]
train_texts = train_df.comment_text.to_numpy()
tokenizer = Tokenizer(num_words=len(train_texts))
tokenizer.fit_on_texts(train_texts)


@app.route('/predict', methods=['POST'])  # decorator defines the route
def predict():
    text = [request.form['comment']]
    print(text)
    string_sequences = tokenizer.texts_to_sequences(text)
    print(string_sequences)
    string_padded = pad_sequences(
        string_sequences, maxlen=50, padding="post", truncating="post")
    print(string_padded)
    # load json and create model
    json_file = open('MCCNN/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("MCCNN/model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='adam', metrics=['accuracy'])
    score = loaded_model.predict([string_padded, string_padded, string_padded])
    result = round((score[0][0]*100), 2)
    return render_template('main.html', prediction = result)


@app.route('/')  # decorator defines the route
def home():
    return render_template('main.html')


if __name__ == '__main__':
    app.run(debug=True)
