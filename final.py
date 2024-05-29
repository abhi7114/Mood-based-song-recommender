# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:29:22 2024

@author: LENOVO
"""

import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

import tkinter as tk
from tkinter import ttk

data=pd.read_csv('spotify_dataset.csv')

# Constants and paths
DETECTION_MODEL_PATH = 'haarcascade_frontalface_default.xml'
EMOTION_MODEL_PATH = "model_weights.h5"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def load_emotion_model():
    # Load emotion model
    try:
        with open('model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model_weights.h5")
        return loaded_model
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        return None

def emotion_testing():
    face_detection = cv2.CascadeClassifier(DETECTION_MODEL_PATH)
    emotion_classifier = load_emotion_model()

    if emotion_classifier is None:
        return

    cap = cv2.VideoCapture(0)

    while True:
        ret, test_img = cap.read()

        if not ret:
            continue

        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_detection.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            fc = gray_img[y:y + h, x:x + w]
            roi_gray = cv2.resize(fc, (48, 48))
            
            prediction = emotion_classifier.predict(roi_gray[np.newaxis, :, :, np.newaxis])
            pred = EMOTIONS[np.argmax(prediction)]

            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), 7)
            cv2.putText(test_img, pred, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial Emotion Analysis', resized_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return pred

#emotion_word=emotion_testing()
#print(emotion_word)
def get_results(emotion_word):
  NUM_RECOMMEND=10
  df = pd.DataFrame(data)
  happy_set=[]
  sad_set=[]
  calm_set=[]
  energetic_set=[]
  
  if emotion_word=="Happy":
      happy_set.append(df[df['mood']=="Happy"]['name'].sample(frac=1).reset_index(drop=True).head(NUM_RECOMMEND))
      return pd.DataFrame(happy_set).T
  elif emotion_word=="Sad":
      sad_set.append(df[df['mood']=="Sad"]['name'].sample(frac=1).reset_index(drop=True).head(NUM_RECOMMEND))
      return pd.DataFrame(sad_set).T
  elif emotion_word=="Disgust" or emotion_word=="Angry" or emotion_word=="Fear":
      calm_set.append(df[df['mood']=="Calm"]['name'].sample(frac=1).reset_index(drop=True).head(NUM_RECOMMEND))
      return pd.DataFrame(calm_set).T
  elif emotion_word=="Neutral" or emotion_word=="Surprise":
      energetic_set.append(df[df['mood']=="Energetic"]['name'].sample(frac=1).reset_index(drop=True).head(NUM_RECOMMEND))
      return pd.DataFrame(energetic_set).T



#print(get_results(emotion_testing()))

# Function to process output and present in a table
def process_and_display(emotion):
    result_list = get_results(emotion)
    output_text.delete("1.0", tk.END)
    for _, row in result_list.iterrows():  # Iterate over rows
        song_name = row['name']  # Extract the 'name' value
        output_text.insert(tk.END, song_name + "\n")

def process_output():
    emotion = emotion_testing()
    if emotion:
        emotion_label.config(text=f"Emotion detected: {emotion}")
        process_and_display(emotion)
        


# Define the GUI elements
root = tk.Tk()
root.title("Text Emotion Analyzer")


# Create a heading label with a larger font
heading = tk.Label(root, text="MOOD BASED SONG RECOMMENDER", font=("Arial", 24))
heading.pack()

# Create a label for regular text
regular_text = "Press q , once emotion is detected"
regular_label = tk.Label(root, text=regular_text)

regular_label.pack()

# Create a frame to hold the widgets
frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

# Create a label for the listbox
listbox_label = ttk.Label(frame, text="List of songs:",font=("Arial", 12))
listbox_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)


listbox = tk.Listbox(frame, width=50)
listbox.grid(row=1, column=0, padx=5, pady=5)

# Populate the listbox with emotions
for text in data['name'].sample(frac=1).reset_index(drop=True):
    listbox.insert(tk.END, text)

# Create a button to analyze the selected emotion
button = ttk.Button(frame, text="Analyze", command=process_output)
button.grid(row=2, column=0, padx=5, pady=5)

# Create a label for the output text
output_text_label = ttk.Label(frame, text="Recommended For You:",font=("Arial", 12))
output_text_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

# Create a text widget to display the output
output_text = tk.Text(frame, width=50, height=10)
output_text.grid(row=1, column=1, padx=5, pady=5)


# Create a label to display the detected emotion
emotion_label = tk.Label(frame, text="Emotion detected: ", font=("Arial", 12))
emotion_label.grid(row=2, column=1, padx=5, pady=5)

# Create a label for the image
image_label = ttk.Label(frame)
image_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Load the image
image = tk.PhotoImage(file="s_image1.png")

# Create a label to display the image
image_label = ttk.Label(frame, image=image)
image_label.image = image
image_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()

