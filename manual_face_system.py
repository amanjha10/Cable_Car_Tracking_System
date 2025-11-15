import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os
from insightface.app import FaceAnalysis
from PIL import Image

# CONFIG
EMBEDDINGS_PKL = "embeddings/faces.pkl"
PROFILES_CSV = "profiles.csv"       
ATTENDANCE_CSV = "attendance.csv"   
SIMILARITY_THRESHOLD = 0.4   

class ManualFaceSystem:
    def __init__(self, similarity_threshold=SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        
        # InsightFace
        print("Loading InsightFace model...")
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Model ready.")

        # Database
        self.db = {}
        self._load_embeddings()
        self._init_csvs()

    def _init_csvs(self):
        if not os.path.exists(PROFILES_CSV):
            pd.DataFrame(columns=['person_id','name','age','location','created_at','status']).to_csv(PROFILES_CSV, index=False)
        if not os.path.exists(ATTENDANCE_CSV):
            pd.DataFrame(columns=['person_id','name','checkin','checkout','status']).to_csv(ATTENDANCE_CSV, index=False)

    def _load_embeddings(self):
        if os.path.exists(EMBEDDINGS_PKL):
            try:
                with open(EMBEDDINGS_PKL, 'rb') as f:
                    self.db = pickle.load(f)
                print(f"Loaded {len(self.db)} profiles from embeddings.")
            except Exception as e:
                print("Failed to load embeddings:", e)
                self.db = {}
        else:
            self.db = {}

    def _save_embeddings(self):
        try:
            with open(EMBEDDINGS_PKL, 'wb') as f:
                pickle.dump(self.db, f)
        except Exception as e:
            print("Failed to save embeddings:", e)

    def _match_embedding(self, embedding):
        """Match embedding against database"""
        best_match = None
        best_score = 0.0
        
        for person_id, info in self.db.items():
            embeddings = info.get('embeddings', [])
            for stored_emb in embeddings:
                score = float(np.dot(embedding, stored_emb))
                if score > best_score and score >= self.similarity_threshold:
                    best_score = score
                    best_match = person_id
        
        return best_match, best_score

    def _update_profile_status(self, person_id, new_status):
        """Update person status in database and CSV"""
        if person_id not in self.db:
            return
        
        self.db[person_id]['status'] = new_status
        self._save_embeddings()

        try:
            df = pd.read_csv(PROFILES_CSV)
            df.loc[df['person_id'] == person_id, 'status'] = new_status
            df.to_csv(PROFILES_CSV, index=False)
        except Exception as e:
            print("Failed to update profiles.csv:", e)

    def _log_attendance_event(self, person_id, person_name, event_type):
        """Log attendance event"""
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            df = pd.read_csv(ATTENDANCE_CSV)
        except Exception:
            df = pd.DataFrame(columns=['person_id','name','checkin','checkout','status'])

        if event_type == "IN":
            new = {'person_id': person_id, 'name': person_name, 'checkin': ts, 'checkout': '', 'status': 'IN'}
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
            print(f"🔵 [{ts}] IN: {person_name} ({person_id})")
        elif event_type == "OUT":
            # Find last row for person with empty checkout
            mask = (df['person_id'] == person_id) & (df['checkout'].astype(str) == '')
            if mask.any():
                idx = df[mask].index[-1]
                df.loc[idx, 'checkout'] = ts
                df.loc[idx, 'status'] = 'OUT'
                print(f"🟠 [{ts}] OUT: {person_name} ({person_id})")
            else:
                # No open IN record -> create OUT only
                new = {'person_id': person_id, 'name': person_name, 'checkin': '', 'checkout': ts, 'status': 'OUT'}
                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                print(f"🟠 [{ts}] OUT(direct): {person_name} ({person_id})")

        df.to_csv(ATTENDANCE_CSV, index=False)

    def register_profile(self, image_path, name, age, location):
        """Register new profile from image file"""
        # Generate person_id
        person_id = "P" + datetime.now().strftime("%Y%m%d%H%M%S")
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load image
        try:
            img = cv2.imread(image_path)
            if img is None:
                # Try PIL for other formats
                pil_img = Image.open(image_path).convert("RGB")
                img = np.array(pil_img)[:, :, ::-1]  # RGB to BGR
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return None

        # Detect faces and select best one
        faces = self.app.get(img)
        if not faces:
            print("No face detected in registration image.")
            return None

        if len(faces) > 1:
            print(f"Multiple faces detected ({len(faces)}), selecting most prominent...")
            def face_score(face):
                bbox = face.bbox
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                confidence = face.det_score
                return area * confidence
            
            faces = sorted(faces, key=face_score, reverse=True)

        face = faces[0]
        emb = face.normed_embedding.astype(np.float32)

        # Store in database
        self.db[person_id] = {
            "name": name,
            "age": age,
            "location": location,
            "created_at": created_at,
            "status": "REGISTERED",
            "embeddings": [emb]
        }
        self._save_embeddings()

        # Add to profiles.csv
        df = pd.read_csv(PROFILES_CSV)
        df = pd.concat([df, pd.DataFrame([{
            'person_id': person_id,
            'name': name,
            'age': age,
            'location': location,
            'created_at': created_at,
            'status': 'REGISTERED'
        }])], ignore_index=True)
        df.to_csv(PROFILES_CSV, index=False)

        print(f"✅ Registered {name} as {person_id}")
        return person_id

    def simulate_detection(self, person_id, camera_type='in'):
        """Manually simulate a detection for testing"""
        if person_id not in self.db:
            print(f"❌ Person ID {person_id} not found in database")
            return False
            
        person_name = self.db[person_id]['name']
        
        # Log attendance based on camera
        if camera_type == 'in':
            self._update_profile_status(person_id, "IN")
            self._log_attendance_event(person_id, person_name, "IN")
        else:  # camera_type == 'out'
            self._update_profile_status(person_id, "OUT")
            self._log_attendance_event(person_id, person_name, "OUT")
            
        print(f"✅ Simulated {camera_type.upper()} detection for {person_name}")
        return True

    def get_camera_status(self):
        """Return dummy camera status for compatibility"""
        return {
            'in': {'mode': 'manual', 'active': True, 'reconnect_attempts': 0, 'last_frame': 0},
            'out': {'mode': 'manual', 'active': True, 'reconnect_attempts': 0, 'last_frame': 0}
        }

    def start(self):
        """Start in manual mode (no actual camera processing)"""
        print("🎯 Face detection system started in MANUAL MODE")
        print("   - RTSP cameras are not accessible")
        print("   - Use 'Test Detection' buttons in web interface")
        print("   - System ready for manual testing")
        return True

    def stop(self):
        """Stop the system"""
        self._save_embeddings()
        print("✅ Manual face detection system stopped.")

# For backward compatibility with existing Flask app
class MultiGateFaceSystem(ManualFaceSystem):
    """Wrapper for backward compatibility"""
    def __init__(self, **kwargs):
        # Ignore camera-related parameters for manual mode
        super().__init__(similarity_threshold=kwargs.get('similarity_threshold', SIMILARITY_THRESHOLD))

if __name__ == "__main__":
    system = ManualFaceSystem()
    system.start()
    
    print("\n" + "="*60)
    print("🎯 MANUAL FACE DETECTION SYSTEM")
    print("="*60)
    print("Available commands:")
    print("  't <person_id>' - Test IN detection")
    print("  'o <person_id>' - Test OUT detection")
    print("  'q' - Quit")
    print("="*60)
    
    # Interactive mode
    while True:
        try:
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd.startswith('t '):
                person_id = cmd.split(' ', 1)[1]
                system.simulate_detection(person_id, 'in')
            elif cmd.startswith('o '):
                person_id = cmd.split(' ', 1)[1]
                system.simulate_detection(person_id, 'out')
            else:
                print("Invalid command. Use 't <id>', 'o <id>', or 'q'")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    system.stop()
