from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import librosa

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Define the mapping from class indices to class names
class_labels = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # Take mean of MFCCs across time frames
    return mfccs_mean

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        audio_file = request.files['file']
        audio_file.save('temp.wav') 
        
        # Extract features from the audio file
        features = extract_features('temp.wav')
        
        # Ensure features are reshaped correctly for prediction
        features_reshaped = features.reshape(1, -1)  
        
        # Make prediction
        prediction = model.predict(features_reshaped)
        
        # Check if prediction is an array of probabilities
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
            predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get index of max probability
        else:
            predicted_class_index = int(prediction[0])  # If it's a single class output
            
        # Map index to class name
        predicted_class_name = class_labels.get(int(predicted_class_index), "Unknown")  # Convert to int
        
        return jsonify({'class_index': int(predicted_class_index), 'class_name': predicted_class_name})  
  



if __name__ == '__main__':
    app.run(debug=True)
