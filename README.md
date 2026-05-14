# Hand Gesture Recognition with Text-to-Speech

This project detects hand gestures in real time using a webcam, recognizes the gesture with a trained Random Forest classifier, and speaks the predicted character aloud using text-to-speech.

The system uses MediaPipe to extract hand landmarks, OpenCV for webcam input and display, scikit-learn for model training, and pyttsx3 for offline speech output.

## Features

- Real-time hand gesture recognition using a webcam
- Hand landmark detection with MediaPipe Hands
- Random Forest based gesture classification
- Text-to-speech output for recognized gestures
- Bounding box, predicted label, and confidence score displayed on screen
- Confidence filtering to reduce incorrect speech output

## Recognized Gestures

The current model is designed for 6 gesture classes:

| Class | Label |
| --- | --- |
| 0 | A |
| 1 | B |
| 2 | L |
| 3 | C |
| 4 | S |
| 5 | 7 |

You can update these labels in `Gesture_recognise.py` if you train the model with different gestures.

## Project Structure

```text
Hand-gesture-recognition/
|-- collect_imgs.py        # Captures webcam images for each gesture class
|-- collect_dataset.py     # Converts saved images into hand landmark data
|-- train_classifier.py    # Trains the Random Forest classifier
|-- Gesture_recognise.py   # Runs real-time gesture recognition and speech
`-- README.md
```

Generated files and folders:

```text
data/          # Created after running collect_imgs.py
data.pickle    # Created after running collect_dataset.py
model.p        # Created after running train_classifier.py
```

## Requirements

Install Python 3.8 or later, then install the required libraries:

```bash
pip install opencv-python mediapipe scikit-learn numpy pyttsx3 matplotlib
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Trisha-Tyagi/Hand-gesture-recognition.git
   ```

2. Move into the project folder:

   ```bash
   cd Hand-gesture-recognition
   ```

3. Install dependencies:

   ```bash
   pip install opencv-python mediapipe scikit-learn numpy pyttsx3 matplotlib
   ```

## How to Run

### 1. Collect Gesture Images

Run:

```bash
python collect_imgs.py
```

This script opens your webcam and collects images for each gesture class.

- Press `q` when you are ready to start collecting images for a class.
- By default, it collects 100 images for each of 6 classes.
- The images are saved inside the `data/` folder.

You can change these values in `collect_imgs.py`:

```python
number_of_classes = 6
dataset_size = 100
```

### 2. Create the Landmark Dataset

Run:

```bash
python collect_dataset.py
```

This script reads the images from `data/`, detects hand landmarks using MediaPipe, and saves the processed features into `data.pickle`.

### 3. Train the Classifier

Run:

```bash
python train_classifier.py
```

This trains a Random Forest classifier and saves the trained model as `model.p`.

### 4. Start Real-Time Gesture Recognition

Run:

```bash
python Gesture_recognise.py
```

The webcam window will show:

- The detected hand landmarks
- A bounding box around the hand
- The predicted gesture label
- The prediction confidence

If the prediction confidence is greater than 60%, the program speaks the recognized character.

Press `q` to close the webcam window.

## Notes

- Make sure your webcam is connected and accessible.
- Good lighting improves hand detection accuracy.
- Keep your hand clearly visible inside the camera frame.
- The trained model file `model.p` must exist before running `Gesture_recognise.py`.
- If you change the number of classes or gesture labels, update `labels_dict` in `Gesture_recognise.py`.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- scikit-learn
- NumPy
- pyttsx3
- Matplotlib

## Future Improvements

- Add a `requirements.txt` file
- Add more gesture classes
- Save predicted letters into words or sentences
- Improve text-to-speech timing
- Add a GUI for easier use
