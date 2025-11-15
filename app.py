from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, Response
from flask_cors import CORS
import os
import csv
import json
import base64
from datetime import datetime
import pandas as pd
import cv2
import threading
import time
import numpy as np

app = Flask(__name__, template_folder='frontend', static_folder='frontend')
CORS(app)

os.makedirs('images', exist_ok=True)
os.makedirs('embeddings', exist_ok=True)

# Camera streaming variables
camera_frames = {'in': None, 'out': None}
camera_threads = {}
camera_running = {'in': False, 'out': False}

# Initialize face system lazily to avoid startup delay
face_system = None

def get_face_system():
    global face_system
    if face_system is None:
        print("Initializing face detection system with fallback support...")
        from fallback_face_system import MultiGateFaceSystem
        face_system = MultiGateFaceSystem(enable_fallback=True)
        print("Face detection system with fallback ready!")
    return face_system

def start_face_detection():
    """Start the face detection system"""
    try:
        system = get_face_system()
        if not hasattr(system, '_detection_started') or not system._detection_started:
            print("🧠 Starting face detection system...")
            system.start()
            system._detection_started = True
            print("✅ Face detection system started successfully!")
        else:
            print("Face detection system already running")
    except Exception as e:
        print(f"❌ Error starting face detection system: {e}")

def generate_camera_feed(camera_type, rtsp_url):
    """Generate camera frames for streaming"""
    global camera_frames, camera_running
    
    print(f"Starting {camera_type.upper()} camera feed from {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    if not cap.isOpened():
        print(f"Failed to open {camera_type.upper()} camera stream: {rtsp_url}")
        camera_running[camera_type] = False
        return
    
    camera_running[camera_type] = True
    print(f"{camera_type.upper()} camera connected successfully")
    
    frame_count = 0
    while camera_running[camera_type]:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 480))
            
            # Add timestamp and camera label
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"{camera_type.upper()} Camera - {timestamp}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                camera_frames[camera_type] = buffer.tobytes()
        else:
            print(f"Failed to read frame from {camera_type.upper()} camera")
            time.sleep(1)
    
    cap.release()
    print(f"{camera_type.upper()} camera feed stopped")

def start_camera_feeds():
    """Start camera feeds"""
    global camera_threads
    
    # IN Camera (192.168.1.12)
    if not camera_running['in']:
        in_thread = threading.Thread(
            target=generate_camera_feed, 
            args=('in', "rtsp://admin:145628%40@192.168.1.12:554/stream1"),
            daemon=True
        )
        camera_threads['in'] = in_thread
        in_thread.start()
    
    # OUT Camera (192.168.1.5) - Now with correct password encoding
    if not camera_running['out']:
        out_thread = threading.Thread(
            target=generate_camera_feed, 
            args=('out', "rtsp://admin:145628%40@192.168.1.5:554/stream1"),
            daemon=True
        )
        camera_threads['out'] = out_thread
        out_thread.start()

def create_placeholder_frame(text, status="Connecting..."):
    """Create a placeholder frame when camera is not available"""
    # Create a blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame.fill(50)  # Dark gray background
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White text
    thickness = 2
    
    # Calculate text size and position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    cv2.putText(frame, status, (text_x - 50, text_y + 50), 
                font, 0.7, (255, 255, 0), 2)
    
    # Convert to JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes() if ret else b''

@app.route('/')
def index():
    return render_template('index.html')

# Static file routes with cache control
@app.route('/styles.css')
def styles():
    response = send_from_directory('frontend', 'styles.css', mimetype='text/css')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/scripts.js')
def scripts():
    response = send_from_directory('frontend', 'scripts.js', mimetype='application/javascript')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/camera/<camera_type>/stream')
def camera_stream(camera_type):
    """Stream camera feed as MJPEG"""
    def generate():
        while True:
            if camera_type in camera_frames and camera_frames[camera_type] is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + camera_frames[camera_type] + b'\r\n')
            else:
                # Send placeholder frame if camera not available
                if camera_type == 'out':
                    placeholder = create_placeholder_frame(f"{camera_type.upper()} Camera", "Camera Offline")
                else:
                    placeholder = create_placeholder_frame(f"{camera_type.upper()} Camera", "Connecting...")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(0.1)  # ~10 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/start', methods=['POST'])
def start_cameras():
    """Start camera feeds"""
    try:
        start_camera_feeds()
        return jsonify({
            'success': True, 
            'message': 'Camera feeds started',
            'cameras': {
                'in': 'Starting...',
                'out': 'Offline (Connection timeout)'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """Get camera status"""
    return jsonify({
        'success': True,
        'cameras': {
            'in': {
                'running': camera_running['in'],
                'url': "rtsp://admin:145628%40@192.168.1.12:554/stream1",
                'status': 'Connected' if camera_running['in'] else 'Connecting...'
            },
            'out': {
                'running': camera_running['out'],
                'url': "rtsp://admin:145628%40@192.168.1.5:554/stream1",
                'status': 'Connected' if camera_running['out'] else 'Connecting...'
            }
        }
    })

@app.route('/api/register', methods=['POST'])
def register_person():
    try:
        data = request.json
        
        # Extract data
        name = data.get('name')
        age = data.get('age')
        location = data.get('location')
        photo_data = data.get('photo')  # base64 encoded image
        
        if not all([name, age, location, photo_data]):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400
        
        # Save photo with improved handling
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name.replace(' ', '_')}_{timestamp}.jpg"
        filepath = os.path.join('images', filename)
        
        # Decode base64 image with validation
        try:
            if 'data:image' in photo_data:
                # Remove data URL prefix (data:image/jpeg;base64, or similar)
                photo_data = photo_data.split(',')[1]
            
            # Clean up base64 data
            photo_data = photo_data.replace(' ', '+')  # Fix any spaces that became +
            
            # Add padding if needed
            missing_padding = len(photo_data) % 4
            if missing_padding:
                photo_data += '=' * (4 - missing_padding)
            
            # Decode base64
            image_data = base64.b64decode(photo_data)
            
            # Validate and convert the image using PIL
            from PIL import Image
            import io
            
            # Open image from bytes
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed (removes alpha channel if present)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Save as JPEG with PIL (this ensures proper format)
            pil_image.save(filepath, 'JPEG', quality=95)
            
            print(f"Successfully saved image: {filepath} (size: {os.path.getsize(filepath)} bytes)")
            
        except Exception as img_error:
            print(f"Image processing error: {img_error}")
            return jsonify({'success': False, 'message': f'Invalid image data: {str(img_error)}'}), 400
        
        # Register with face system (lazy loading)
        system = get_face_system()
        person_id = system.register_profile(filepath, name, int(age), location)
        
        if person_id:
            return jsonify({
                'success': True, 
                'message': f'Successfully registered {name}',
                'person_id': person_id
            })
        else:
            return jsonify({'success': False, 'message': 'No face detected in image'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/people', methods=['GET'])
def get_people():
    try:
        # Read from profiles.csv
        if os.path.exists('profiles.csv'):
            df = pd.read_csv('profiles.csv')
            people = []
            
            for _, row in df.iterrows():
                # Get attendance info
                attendance_info = get_person_attendance(row['person_id'])
                
                person = {
                    'id': row['person_id'],
                    'name': row['name'],
                    'age': row['age'],
                    'location': row['location'],
                    'status': row['status'],
                    'created_at': row['created_at'],
                    'last_in': attendance_info.get('last_in', ''),
                    'last_out': attendance_info.get('last_out', ''),
                    'photo': get_person_photo(row['name'])
                }
                people.append(person)
            
            return jsonify({'success': True, 'people': people})
        else:
            return jsonify({'success': True, 'people': []})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def get_person_attendance(person_id):
    try:
        if os.path.exists('attendance.csv'):
            df = pd.read_csv('attendance.csv')
            person_records = df[df['person_id'] == person_id]
            
            if not person_records.empty:
                last_in = ''
                last_out = ''
                
                # Get last check-in
                checkins = person_records[person_records['checkin'].notna() & (person_records['checkin'] != '')]
                if not checkins.empty:
                    last_in = checkins.iloc[-1]['checkin']
                
                # Get last check-out
                checkouts = person_records[person_records['checkout'].notna() & (person_records['checkout'] != '')]
                if not checkouts.empty:
                    last_out = checkouts.iloc[-1]['checkout']
                
                return {'last_in': last_in, 'last_out': last_out}
    except:
        pass
    
    return {'last_in': '', 'last_out': ''}

def get_person_photo(name):
    # Find photo file for person
    image_dir = 'images'
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if name.replace(' ', '_') in filename:
                return f'/api/photo/{filename}'
    return None

@app.route('/api/photo/<filename>')
def get_photo(filename):
    filepath = os.path.join('images', filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return '', 404

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        total_registered = 0
        total_in = 0
        total_out = 0
        
        # Count from profiles.csv
        if os.path.exists('profiles.csv'):
            df = pd.read_csv('profiles.csv')
            total_registered = len(df)
            total_in = len(df[df['status'] == 'IN'])
            total_out = len(df[df['status'] == 'OUT'])
        
        return jsonify({
            'success': True,
            'stats': {
                'total_registered': total_registered,
                'total_in': total_in,
                'total_out': total_out
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/system/start', methods=['POST'])
def start_system():
    try:
        # Initialize face system and start it
        system = get_face_system()
        
        # Start face detection system in background thread
        def run_system():
            try:
                system.start()
            except Exception as e:
                print(f"Face detection system error: {e}")
        
        system_thread = threading.Thread(target=run_system, daemon=True)
        system_thread.start()
        
        return jsonify({'success': True, 'message': 'Face detection system started'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    try:
        if os.path.exists('profiles.csv'):
            return send_file('profiles.csv', as_attachment=True, download_name='tracking_data.csv')
        else:
            return jsonify({'success': False, 'message': 'No data to export'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/test-image', methods=['POST'])
def test_image():
    """Test endpoint to validate image upload and face detection"""
    try:
        data = request.json
        photo_data = data.get('photo')
        
        if not photo_data:
            return jsonify({'success': False, 'message': 'No photo data provided'}), 400
        
        # Process image like in registration
        if 'data:image' in photo_data:
            photo_data = photo_data.split(',')[1]
        
        try:
            # Decode base64
            image_data = base64.b64decode(photo_data)
            
            # Test with PIL
            from PIL import Image
            import io
            test_image = Image.open(io.BytesIO(image_data))
            
            # Test face detection
            import numpy as np
            import cv2
            
            # Convert PIL to OpenCV format
            img_array = np.array(test_image)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            # Initialize face system and detect faces
            system = get_face_system()
            faces = system.app.get(img_cv)
            
            return jsonify({
                'success': True,
                'message': 'Image processed successfully',
                'details': {
                    'image_size': test_image.size,
                    'image_mode': test_image.mode,
                    'faces_detected': len(faces),
                    'data_size': len(image_data)
                }
            })
            
        except Exception as img_error:
            return jsonify({
                'success': False, 
                'message': f'Image processing failed: {str(img_error)}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/people/<person_id>/delete', methods=['DELETE'])
def delete_person(person_id):
    """Delete a registered person"""
    try:
        # Read current profiles
        if not os.path.exists('profiles.csv'):
            return jsonify({'success': False, 'message': 'No profiles found'}), 404
        
        df = pd.read_csv('profiles.csv')
        person_row = df[df['person_id'] == person_id]
        
        if person_row.empty:
            return jsonify({'success': False, 'message': 'Person not found'}), 404
        
        person_name = person_row.iloc[0]['name']
        
        # Remove from profiles.csv
        df = df[df['person_id'] != person_id]
        df.to_csv('profiles.csv', index=False)
        
        # Remove from attendance.csv if exists
        if os.path.exists('attendance.csv'):
            att_df = pd.read_csv('attendance.csv')
            att_df = att_df[att_df['person_id'] != person_id]
            att_df.to_csv('attendance.csv', index=False)
        
        # Remove from embeddings
        system = get_face_system()
        if person_id in system.db:
            del system.db[person_id]
            system._save_embeddings()
        
        # Remove image files
        import glob
        image_pattern = f"images/{person_name.replace(' ', '_')}_*.jpg"
        for image_file in glob.glob(image_pattern):
            try:
                os.remove(image_file)
                print(f"Deleted image: {image_file}")
            except Exception as e:
                print(f"Error deleting image {image_file}: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully deleted {person_name} and all associated data'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/people/<person_id>/generate-embedding', methods=['POST'])
def generate_embedding(person_id):
    """Generate face embedding for a registered person"""
    try:
        # Read current profiles
        if not os.path.exists('profiles.csv'):
            return jsonify({'success': False, 'message': 'No profiles found'}), 404
        
        df = pd.read_csv('profiles.csv')
        person_row = df[df['person_id'] == person_id]
        
        if person_row.empty:
            return jsonify({'success': False, 'message': 'Person not found'}), 404
        
        person_name = person_row.iloc[0]['name']
        
        # Find the person's image file
        import glob
        image_pattern = f"images/{person_name.replace(' ', '_')}_*.jpg"
        image_files = glob.glob(image_pattern)
        
        if not image_files:
            return jsonify({'success': False, 'message': 'No image file found for this person'}), 404
        
        # Use the most recent image file
        image_file = max(image_files, key=os.path.getctime)
        
        # Check if image exists and is readable
        if not os.path.exists(image_file):
            return jsonify({'success': False, 'message': 'Image file not found'}), 404
        
        # Test if image can be opened by OpenCV
        import cv2
        test_img = cv2.imread(image_file)
        if test_img is None:
            return jsonify({'success': False, 'message': 'Image file is corrupted or unreadable'}), 400
        
        # Generate embedding using face system
        system = get_face_system()
        
        # Get faces from image
        faces = system.app.get(test_img)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected in the image'}), 400
        
        # If multiple faces, select the largest/most prominent one
        if len(faces) > 1:
            print(f"Multiple faces detected ({len(faces)}), selecting the most prominent one...")
            # Sort faces by area (width * height) and confidence
            def face_score(face):
                bbox = face.bbox
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                confidence = face.det_score
                return area * confidence  # Combine area and confidence
            
            faces = sorted(faces, key=face_score, reverse=True)
            print(f"Selected face with score: {face_score(faces[0]):.2f}")
        
        # Extract embedding from the best face
        face = faces[0]
        embedding = face.embedding
        
        # Store embedding in the database
        if person_id not in system.db:
            system.db[person_id] = {
                'name': person_name,
                'embeddings': [],
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Add the embedding
        system.db[person_id]['embeddings'] = [embedding]
        system._save_embeddings()
        
        print(f"Generated embedding for {person_name} (ID: {person_id})")
        
        return jsonify({
            'success': True,
            'message': f'Successfully generated face embedding for {person_name}',
            'details': {
                'person_id': person_id,
                'person_name': person_name,
                'image_file': os.path.basename(image_file),
                'embedding_dimensions': len(embedding)
            }
        })
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/people/<person_id>/check-embedding', methods=['GET'])
def check_embedding(person_id):
    """Check if a person has an embedding"""
    try:
        system = get_face_system()
        has_embedding = person_id in system.db and len(system.db[person_id].get('embeddings', [])) > 0
        
        return jsonify({
            'success': True,
            'has_embedding': has_embedding,
            'person_id': person_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/detections/recent', methods=['GET'])
def get_recent_detections():
    """Get recent face detections and attendance updates"""
    try:
        # Read recent attendance records (last 10)
        recent_detections = []
        
        if os.path.exists('attendance.csv'):
            df = pd.read_csv('attendance.csv')
            # Replace NaN values with empty strings
            df = df.fillna('')
            # Get last 10 records, sorted by most recent
            recent_records = df.tail(10).to_dict('records')
            
            for record in recent_records:
                checkin = str(record.get('checkin', '')).replace('nan', '')
                checkout = str(record.get('checkout', '')).replace('nan', '')
                detection = {
                    'person_id': str(record.get('person_id', '')),
                    'name': str(record.get('name', '')),
                    'checkin': checkin if checkin != 'nan' else '',
                    'checkout': checkout if checkout != 'nan' else '',
                    'status': str(record.get('status', '')),
                    'timestamp': checkin if checkin else checkout
                }
                recent_detections.append(detection)
        
        # Also get current status of all people
        people_status = {}
        if os.path.exists('profiles.csv'):
            profiles_df = pd.read_csv('profiles.csv')
            for _, row in profiles_df.iterrows():
                people_status[row['person_id']] = {
                    'name': row['name'],
                    'status': row['status']
                }
        
        return jsonify({
            'success': True,
            'recent_detections': recent_detections,
            'people_status': people_status
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/people/<person_id>/test-detection', methods=['POST'])
def test_detection(person_id):
    """Simulate face detection for testing"""
    try:
        data = request.json or {}
        camera_type = data.get('camera_type', 'in')  # 'in' or 'out'
        
        system = get_face_system()
        success = system.simulate_detection(person_id, camera_type)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Simulated {camera_type.upper()} detection successfully',
                'person_id': person_id,
                'camera_type': camera_type
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Person not found or detection failed'
            }), 404
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/status-detailed', methods=['GET'])
def get_detailed_camera_status():
    """Get detailed camera status including fallback info"""
    try:
        system = get_face_system()
        camera_status = system.get_camera_status()
        
        return jsonify({
            'success': True,
            'cameras': camera_status,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Cable Car Tracking System...")
    print("📷 Camera Status:")
    print("   ✅ IN Camera (192.168.1.12): Available")
    print("   ❌ OUT Camera (192.168.1.5): Offline")
    print("🌐 Frontend available at: http://localhost:8090")
    print("⚡ Initializing face detection system...")
    
    # Start face detection system
    start_face_detection()
    
    # Auto-start camera feeds (for streaming only)
    start_camera_feeds()
    
    app.run(debug=True, host='0.0.0.0', port=8090, threaded=True)
