# Baby Vision AI

An intelligent detection system for baby monitoring using computer vision and MediaPipe. This project is designed to help parents monitor their baby's condition in real-time with various detection features.

## 🌟 Features

- **Eye Detection**: Monitors the baby's eye state (open/closed/half-open) using Eye Aspect Ratio (EAR)
- **Mouth Detection**: Identifies the baby's mouth state (open/closed) using Mouth Aspect Ratio (MAR)
- **Movement Detection**: Monitors the baby's movement activity with adjustable sensitivity
- **Pacifier Detection**: Detects the presence of a pacifier in the baby's mouth
- **Rollover Detection**: Identifies if the baby changes position (rolling over/face down)
- **Enhanced Frame**: Image quality enhancement for more accurate detection at a distance
- **API Endpoint**: REST API for integration with other applications
- **Automatic Annotation**: Pre-labeling system for video datasets

## 📋 Prerequisites

- Python 3.7 or higher
- Webcam (for real-time detection)
- Internet connection (for installing dependencies)

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/username/baby-vision-ai.git
cd baby-vision-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required dependencies:
- fastapi
- uvicorn
- mediapipe
- opencv-python
- numpy
- scipy

## 📖 Usage

### Real-Time Mode (Webcam)

Run real-time detection using webcam:
```bash
python main.py
```

Press the `q` key to exit the application.

### API Server Mode

Run the API server for integration with other applications:
```bash
python app.py
```

The API will run at `http://localhost:8000`

Example API usage:
```bash
curl -X POST "http://localhost:8000/detect" -F "file=@image.jpg"
```

### Dataset Pre-labeling

To create automatic annotations from videos:
```bash
python prelabel.py
```

Annotations will be saved in `dataset/annotations.csv`

## 📁 Project Structure

```
baby-vision-ai/
├── app.py                 # FastAPI application
├── main.py                # Main script for real-time detection
├── prelabel.py            # Script for dataset pre-labeling
├── requirements.txt       # Dependencies
├── dataset/              # Video dataset and annotations
│   ├── annotations.csv
│   └── *.mp4
└── src/
    ├── detector.py       # Core detection and logic
    └── utils.py          # Utility functions
```

## 🔧 Configuration

### Detection Thresholds

You can adjust detection thresholds in `src/dector.py`:

- **EAR Threshold** (Eye Aspect Ratio):
  - Eyes closed: < 0.2
  - Eyes half-open: 0.2 - 0.25
  - Eyes open: > 0.25

- **MAR Threshold** (Mouth Aspect Ratio):
  - Mouth closed: ≤ 0.4
  - Mouth open: > 0.4

- **Movement Threshold**:
  - Default: 0.03
  - Adjust in `detect_movement()` function

- **Rollover Threshold**:
  - Default: 30 pixels
  - Adjust in `detect_rollover()` function

## 🎯 How It Works

### Eye Detection
Uses MediaPipe Face Mesh to detect 468 landmarks on the face. Eye Aspect Ratio (EAR) is calculated from 6 landmark points on each eye to determine the eye state.

### Mouth Detection
Uses 4 landmark points in the mouth area to calculate Mouth Aspect Ratio (MAR). This ratio helps identify whether the baby's mouth is open or closed.

### Movement Detection
Compares body landmark positions between frames to calculate movement. A low threshold is used for high sensitivity to small baby movements.

### Pacifier Detection
Uses a combination of color thresholding and contour detection in the mouth area identified by facial landmarks.

### Rollover Detection
Analyzes the vertical position difference between left-right shoulders and left-right hips to detect changes in the baby's position.

## 📊 Output

The system provides output in the following format:

```json
{
  "eye_status": "Eyes closed [EAR: 0.18]",
  "mouth_status": "Mouth closed [MAR: 0.35]",
  "movement_status": "Moving!",
  "pacifier_status": "Pacifier: In use",
  "rollover_status": "Rollover: No"
}
```

## ⚠️ Important Notes

- Ensure adequate room lighting for accurate detection
- The optimal camera distance is 1-2 meters from the baby
- The system works better when the baby is in a supine position
- Avoid excessive camera movement for stable detection results

## 🤝 Contributing

Contributions are highly appreciated! If you want to contribute:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Developers

- [Developer Name] - Initial work

## 🙏 Acknowledgments

- MediaPipe team for their excellent computer vision library
- OpenCV community
- All contributors who have helped develop this project

## 📞 Contact

If you have any questions or suggestions, please open an issue in the repository or contact via email.
