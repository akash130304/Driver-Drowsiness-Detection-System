# Driver Drowsiness Detection System

## 📌 Overview
This project detects driver drowsiness in real time using computer vision techniques. The system monitors eye closure and yawning patterns through a webcam feed and triggers an alert when signs of fatigue are detected.

## 🚀 Features
- Real-time face and eye detection
- Eye Aspect Ratio (EAR) based drowsiness detection
- Yawn detection using facial landmarks
- Audio alert system for driver warning
- Works under different lighting conditions

## 🛠️ Tech Stack
- Python
- OpenCV
- Dlib 
- NumPy

## ⚙️ How It Works
1. Webcam captures live video.
2. Facial landmarks are detected.
3. Eye Aspect Ratio (EAR) is computed.
4. If EAR falls below threshold → drowsiness detected.
5. Alarm is triggered.

## 📊 Results
- Real-time detection achieved
- Works at ~XX FPS (fill if you know)
- Accuracy: XX% (fill if available)

## ▶️ How to Run

```bash
git clone https://github.com/akash130304/Driver-Drowsiness-Detection-System.git
cd Driver-Drowsiness-Detection-System
pip install -r requirements.txt
python main.py
