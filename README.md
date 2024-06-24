# Sign Language Detection

A real-time application for detecting and recognizing sign language gestures using a webcam feed.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Libraries Used](#libraries-used)
- [Contact](#contact)

## Overview
This project leverages Flask for the web interface and TensorFlow/Keras for the machine learning model to recognize sign language gestures in real-time from a webcam feed.

## Features
- Real-time sign language gesture detection.
- Web interface for video feed and gesture recognition.
- Utilizes a CNN model for gesture recognition.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/RiaanSadiq/Sign-Language-Detection.git
    cd Sign-Language-Detection
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure a webcam is connected to your system.

## Usage
1. Start the Flask application:
    ```bash
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. The web interface will display the webcam feed and detected sign language gestures.

## Libraries Used
- **Flask**: Web framework
- **OpenCV**: Video capture and processing
- **NumPy**: Numerical operations
- **TensorFlow/Keras**: Machine learning model
- **Splitfolders**: Dataset splitting
- **Logging**: Logging messages


## Contact
For questions or suggestions, please open an issue or contact me at [riaansadiqa23@gmail.com].
