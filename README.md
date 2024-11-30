# Blink Detection System

This project implements a blink detection system using OpenCV, Dlib, and imutils. It processes a video file to detect blinks based on the Eye Aspect Ratio (EAR).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Model](#model)
- [License](#license)

## Introduction

The Blink Detection System utilizes facial landmark detection to monitor eye movement in a given video. By calculating the Eye Aspect Ratio (EAR), the program determines whether a blink has occurred based on a predefined threshold. This project can be useful in various applications, including fatigue detection in drivers and eye-tracking studies.

## Installation

To set up the project, clone this repository and install the required packages.

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
```

## Usage
1. Place your video file (e.g., my_blink.mp4) in the assets folder.
2. Download the Dlib shape predictor model (shape_predictor_68_face_landmarks.dat) and place it in the Model folder. You can download it from Dlib's model downloads or you can find it in this repo only.
3. Run the main script:
```bash
python blink_detection.py
```
4. The program will start processing the video. You can press 'q' to exit the program at any time.

## Model
The project uses the shape_predictor_68_face_landmarks.dat model provided by Dlib for facial landmark detection. Ensure that the model file is in the correct path as specified in the code.

## License
This project is licensed under the MIT License - see the [LICENSE](mit-license) file for details.