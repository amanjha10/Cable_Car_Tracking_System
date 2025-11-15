# Cable Car Tracking System - Project Structure

## 📁 Essential Files Only

```
📦 Cable Car Tracking System/
├── 🐍 app.py                    # Main Flask application
├── 🧠 multigate_face_det.py     # Face detection & recognition system
├── 📋 requirements.txt          # Python dependencies
├── 📖 README.md                 # Documentation
├── 📊 profiles.csv              # Person registration data
├── 📊 attendance.csv            # IN/OUT tracking records
├── 🗂️ embeddings/
│   └── faces.pkl                # Face embeddings database
├── 🖼️ images/                   # Person photos storage
│   ├── Test_Person_*.jpg
├── 🌐 frontend/                 # Web interface
│   ├── index.html               # Main dashboard
│   ├── styles.css               # UI styling
│   ├── scripts.js               # Frontend logic
│   └── favicon.ico              # Browser icon
└── 🐍 env/                      # Python virtual environment
    └── (Python 3.11 + all dependencies)
```

## 🚀 Quick Start

1. **Activate Environment**: `source env/bin/activate`
2. **Run System**: `python app.py`
3. **Access Dashboard**: http://localhost:8090

## 🎯 Core Components

- **Flask Backend**: Camera streaming, face detection, API endpoints
- **Frontend**: Responsive web dashboard with live camera feeds
- **Face System**: InsightFace for accurate face recognition
- **Data Storage**: CSV files for person profiles and attendance
- **Camera Integration**: RTSP streaming for IN/OUT cameras

## 📋 Dependencies

All required packages are in `requirements.txt` and installed in `env/`:
- Flask, OpenCV, InsightFace, Pandas, NumPy, etc.

---
*System ready for deployment with minimal, essential files only!*
