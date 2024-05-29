# Mood-Based Song Recommender

This project uses facial emotion recognition to recommend songs based on the user's detected mood. It utilizes a pre-trained emotion detection model and a Spotify dataset to suggest appropriate songs.

## Features

- **Facial Emotion Detection**: Uses OpenCV and a pre-trained model to detect facial emotions.
- **GUI Interface**: Built with Tkinter, provides an intuitive interface for users to interact with the application.
- **Song Recommendations**: Recommends songs based on detected emotions using a Spotify dataset.

## Requirements

The following Python libraries are required to run this project:

- matplotlib==3.8.0
- numpy==1.26.4
- opencv_python==4.9.0.80
- pandas==2.2.2
- seaborn==0.13.2
- tensorflow==2.10.0

Install the required libraries using: pip install -r requirements.txt

Dataset
The Spotify dataset used in this project is available on Kaggle. It contains songs and describes the mood of each song as 'happy', 'sad', 'Calm' or 'Energetic'.
spotify_dataset.csv: The dataset file containing song information.
requirements.txt: File listing the required Python libraries.


How It Works
GUI Interface: When the application starts, a GUI built with Tkinter is displayed. It contains a list of songs and a button to start the analysis.
Facial Emotion Detection: When the "Analyze" button is clicked, the application starts the camera and begins detecting facial emotions using OpenCV and a pre-trained model.
Emotion-Based Song Recommendation: Based on the detected emotion, the application recommends a list of songs from the dataset. The detected emotion and the recommended songs are displayed in the GUI.

Example
Run the application(final.py).
The GUI will display a list of songs.
Click on the "Analyze" button.
The camera will start, and the application will detect your facial emotion.
Once the emotion is detected, press 'q' to stop the camera.
The detected emotion and a list of recommended songs will be displayed in the GUI.

Acknowledgments
The Spotify dataset used in this project is available on Kaggle.
The facial emotion recognition model is pre-trained and available online.

Contact
For any questions or suggestions, please feel free to contact me at abhisingh7114@gmail.com.
