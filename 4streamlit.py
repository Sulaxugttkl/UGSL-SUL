import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import tensorflow as tf
import os
from pydub import AudioSegment
from pydub.generators import Sine
import csv
import time
import pyttsx3

from Commonwords import calc_landmark_list, pre_process_landmark, pre_process_point_history

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load labels from CSV
def load_labels(csv_path='model/keypoint_classifier/keypoint_classifier_label.csv'):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        labels = [row[0] for row in reader]
    return labels

class KeyPointClassifier(object):
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.tflite', num_threads=1):
        print("Loading model from:", model_path)
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        print("Interpreter allocated tensors successfully")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Expected input shape:", self.input_details[0]['shape'])

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        input_shape = self.input_details[0]['shape']
        print("Input shape:", input_shape)
        
        # Ensure landmark_list is a NumPy array and has the correct shape
        landmark_list = np.array(landmark_list, dtype=np.float32).flatten()
        landmark_list = np.expand_dims(landmark_list, axis=0)
        print("Processed landmark_list shape:", landmark_list.shape)
        
        self.interpreter.set_tensor(input_details_tensor_index, landmark_list)
        self.interpreter.invoke()
        
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        print("Predicted class:", result_index)
        return result_index

def generate_audio_from_class(class_index, output_file_path, duration=1000, sample_rate=44100):
    class_frequencies = {0: 440, 1: 523}
    frequency = class_frequencies.get(class_index, 440)
    audio = Sine(frequency).to_audio_segment(duration=duration, volume=-3)
    audio.export(output_file_path, format="wav")

output_dir = 'audio_output'
os.makedirs(output_dir, exist_ok=True)
result_index = 1
audio_file_path = f"audio_output/prediction_{result_index}.wav"
generate_audio_from_class(result_index, audio_file_path)

def process_frame(frame, point_history, predictions, labels, last_speech_time):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_list = calc_landmark_list(image, hand_landmarks)
            print("Landmark List:", landmark_list)
            pre_processed_landmarks = pre_process_landmark(landmark_list)
            print("Pre-processed Landmarks:", pre_processed_landmarks)
            prediction = model(pre_processed_landmarks)
            prediction_label = labels[prediction]
            print("Prediction Label:", prediction_label)
            predictions.append((prediction, time.time()))
            point_history.append(landmark_list[8])
            if len(point_history) > 16:
                point_history.popleft()
            pre_processed_point_history = pre_process_point_history(image, point_history)
            print("Frame processing complete.")
            
            # Remove predictions older than 5 seconds
            current_time = time.time()
            predictions = [(pred, timestamp) for pred, timestamp in predictions if current_time - timestamp <= 5]
            
            # Determine the most common prediction in the last 5 seconds
            if current_time - last_speech_time >= 5:
                if predictions:
                    most_common_pred = Counter([pred for pred, timestamp in predictions]).most_common(1)[0][0]
                    most_common_label = labels[most_common_pred]
                    print("Most Common Gesture in Last 5 Seconds:", most_common_label)
                    engine.say(most_common_label)
                    engine.runAndWait()
                    last_speech_time = current_time

    return image, predictions, last_speech_time

st.title("Hand Gesture Recognition with Streamlit")

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

point_history = deque(maxlen=16)
model = KeyPointClassifier()
labels = load_labels()
predictions = []
last_speech_time = time.time()

# Initialize TTS engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image")
        break
    frame, predictions, last_speech_time = process_frame(frame, point_history, predictions, labels, last_speech_time)
    FRAME_WINDOW.image(frame)

cap.release()
st.write("Stopped")
