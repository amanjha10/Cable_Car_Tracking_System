# Cable Car Tracking System

A real-time face recognition system for tracking people entering and exiting cable car stations using computer vision and AI.

## 🚀 Features

- **Real-time Face Detection**: Uses InsightFace AI models for accurate face recognition
- **Dual Camera System**: Separate IN and OUT camera monitoring
- **Web Dashboard**: Modern, responsive frontend for registration and monitoring
- **Live Status Tracking**: Real-time updates of people's IN/OUT status
- **Photo Registration**: Register people with their photos for face recognition
- **Data Export**: Export tracking data to CSV format
- **Status Management**: Three-state tracking (REGISTERED/RED, IN/YELLOW, OUT/GREEN)

## 🛠️ System Requirements

- Python 3.8+
- OpenCV compatible cameras or RTSP streams
- Minimum 4GB RAM (8GB recommended)
- GPU support optional (CUDA/Apple Silicon)

## 📋 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "/Users/amanjha/Documents/Tracking System"
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Make startup script executable:**
   ```bash
   chmod +x start_system.sh
   ```

## 🎯 Quick Start

### Method 1: Using Startup Script
```bash
./start_system.sh
```

### Method 2: Manual Start
```bash
source env/bin/activate
python app.py
```

The system will be available at: **http://localhost:5001**

## 🏗️ System Architecture

```
📁 Project Structure
├── app.py                    # Flask backend server
├── multigate_face_det.py    # Face detection system
├── frontend/                # Web interface files
│   ├── index.html          # Main dashboard
│   ├── styles.css          # UI styling
│   └── scripts.js          # Frontend logic
├── images/                 # Stored user photos
├── embeddings/            # Face embeddings database
├── profiles.csv          # User profiles data
├── attendance.csv        # IN/OUT tracking logs
└── requirements.txt      # Python dependencies
```

## 🔧 Configuration

### Camera Settings (multigate_face_det.py)
```python
IN_RTSP = "rtsp://admin:password@192.168.1.12:554/stream1"   # IN Camera
OUT_RTSP = "rtsp://admin:password@192.168.1.5:554/stream1"   # OUT Camera
SIMILARITY_THRESHOLD = 0.4    # Face recognition sensitivity
DEBOUNCE_SECONDS = 3         # Prevent duplicate detections
TARGET_FPS = 8               # Processing frame rate
```

### Server Settings (app.py)
```python
HOST = '0.0.0.0'            # Server host
PORT = 5001                 # Server port
DEBUG = True                # Development mode
```

## 📷 Camera Configuration

The system is configured to work with two RTSP cameras:

### Camera Details
- **IN Camera**: `192.168.1.12:554` - Monitors people entering
- **OUT Camera**: `192.168.1.5:554` - Monitors people exiting
- **Username**: admin
- **Password**: 145628@
- **Stream Path**: /stream1

### RTSP URLs (as configured in multigate_face_det.py):
```python
IN_RTSP = "rtsp://admin:14562%40@192.168.1.12:554/stream1"
OUT_RTSP = "rtsp://admin:14562%40@192.168.1.5:554/stream1"
```

### Testing Camera Connection
Use the camera test application to verify RTSP connections:
```bash
python camera_test.py
```

This will start a test server at http://localhost:8081 showing live camera feeds.

## 📊 Usage Workflow

1. **Register People**:
   - Click "Register Person" button
   - Upload/capture photo
   - Enter name, age, and location
   - System creates face embeddings automatically

2. **Monitor IN/OUT**:
   - Camera 1 (IN): Detects people entering → Status: YELLOW
   - Camera 2 (OUT): Detects people exiting → Status: GREEN
   - Real-time dashboard updates show current status

3. **Track & Export**:
   - View live statistics on dashboard
   - Search registered people
   - Export data to CSV for reporting

## 🎨 Status System

| Status | Color | Description |
|--------|-------|-------------|
| REGISTERED | 🔴 RED | Person registered but not yet detected |
| IN | 🟡 YELLOW | Person detected entering (Camera 1) |
| OUT | 🟢 GREEN | Person detected exiting (Camera 2) |

## 📁 Data Storage

- **profiles.csv**: Person details (ID, name, age, location, status)
- **attendance.csv**: IN/OUT timestamps and events
- **embeddings/faces.pkl**: Face recognition data (binary)
- **images/**: Stored user photos (JPG format)

## 🔌 API Endpoints

- `GET /` - Web dashboard
- `POST /api/register` - Register new person
- `GET /api/people` - Get all registered people
- `GET /api/stats` - Get system statistics
- `GET /api/export/csv` - Export data
- `POST /api/system/start` - Start face detection
- `POST /api/system/stop` - Stop face detection

## 🚨 Troubleshooting

### Common Issues:

1. **Port 5000 in use**:
   - System automatically uses port 5001
   - Disable macOS AirPlay Receiver if needed

2. **Camera connection failed**:
   - Check RTSP URLs in `multigate_face_det.py`
   - Verify camera credentials and network access

3. **Face detection not working**:
   - Ensure good lighting conditions
   - Check camera positioning and angles
   - Verify InsightFace model installation

4. **Performance issues**:
   - Reduce `TARGET_FPS` for slower systems
   - Use CPU-only mode if GPU causes issues
   - Adjust `SIMILARITY_THRESHOLD` for accuracy

## 📱 Browser Compatibility

- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+

## ⚡ Performance Tips

- Use GPU acceleration when available
- Position cameras at eye level for better detection
- Ensure consistent lighting conditions
- Register multiple photos per person for better accuracy

## 🔒 Security Notes

- System stores face embeddings, not actual photos (for privacy)
- Use HTTPS in production environments
- Secure RTSP camera credentials
- Regular backup of CSV data files

## 📞 Support

For technical issues or feature requests, check the system logs in the terminal where the Flask server is running.

---

**Developed by Spell Innovation** 🚀  
*Advanced Cable Car Tracking Solution*
