import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
from keras.models import load_model
from flask import Flask, jsonify, request

app = Flask(__name__)

model = load_model('model.h5')

with open('text_model.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Tải giá trị của max_sequence_length
with open('max_sequence_length.pkl', 'rb') as f:
    max_sequence_length = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/api/predict', methods=['POST'])
def predict_project():
    description = [request.json['description']]
    print(len(description[0]))
    # Tiền xử lý dữ liệu dự đoán
    new_sequences = tokenizer.texts_to_sequences(description)
    new_data = pad_sequences(new_sequences, maxlen=max_sequence_length)

    print("new_sequences", len(new_sequences[0]))
    print("sequence_length", len(new_data[0]))

    # Dự đoán category cho dữ liệu mới
    predictions = model.predict(new_data)

    # Giải mã kết quả dự đoán thành nhãn
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    result = {'category' : int(predicted_labels[0])}
    return result, 200

@app.route('/', methods=['GET'])
def hello():
    return "Hello"

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0')