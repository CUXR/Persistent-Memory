import os
import sys
import time
import uuid
from datetime import datetime, timezone
import numpy as np
import cv2

import faiss
from insightface.app import FaceAnalysis
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, Session

# Import backend models and config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../backend'))
from app.core.database import SessionLocal, Base
from app.core.config import get_settings
from app.models.person import Person
from app.models.user import User

settings = get_settings()

# Configuration
L2_DIST_THRESHOLD = 1.0
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
MODEL_IDENTIFIER = f"{MODEL_NAME}_insightface"

# Get user ID from environment or use default
# In production, you'd get this from authentication
USER_ID = os.getenv("USER_ID")


def get_user_id(db: Session) -> uuid.UUID:
    """Get or create a default user for face recognition."""
    global USER_ID
    
    if USER_ID:
        try:
            return uuid.UUID(USER_ID)
        except ValueError:
            print(f"Invalid USER_ID format: {USER_ID}")
            sys.exit(1)
    
    # Try to get first user or create one
    user = db.query(User).first()
    if user:
        return user.id
    
    # Create a default user if none exists
    default_user = User(
        first_name="Local",
        last_name="User",
        username="local_user",
        display_name="Local Face Recognition User"
    )
    db.add(default_user)
    db.commit()
    db.refresh(default_user)
    return default_user.id


def load_all_embeddings(db: Session, user_id: uuid.UUID, dim: int):
    """
    Load all face embeddings for a user from the backend database.
    
    Returns:
      vectors: (N, dim) float32
      person_ids: (N,) UUID
      id_to_display_name: dict UUID -> display_name
    """
    # Query all people with face embeddings for this user
    people = db.query(Person).filter(
        and_(
            Person.user_id == user_id,
            Person.face_embedding.isnot(None)
        )
    ).all()
    
    if not people:
        return np.empty((0, dim), dtype=np.float32), np.empty((0,), dtype=object), {}
    
    vectors = []
    person_ids = []
    id_to_display_name = {}
    
    for person in people:
        if person.face_embedding is not None:
            # pgvector returns embeddings as lists
            vec = np.array(person.face_embedding, dtype=np.float32).reshape(1, -1)
            vectors.append(vec[0])
            person_ids.append(person.id)
            display_name = person.display_name or f"{person.first_name} {person.last_name}"
            id_to_display_name[person.id] = display_name
    
    if not vectors:
        return np.empty((0, dim), dtype=np.float32), np.empty((0,), dtype=object), {}
    
    vectors = np.vstack(vectors).astype(np.float32)
    person_ids = np.array(person_ids, dtype=object)
    return vectors, person_ids, id_to_display_name

class FaceRecognizer:
    def __init__(self, db: Session, user_id: uuid.UUID):
        self.db = db
        self.user_id = user_id

        self.app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=DET_SIZE)

        # Determine embedding dimension from settings
        self.dim = settings.embedding_dimension

        self.index = None
        self.person_ids = np.empty((0,), dtype=object)  # aligned to index rows, stores UUIDs
        self.id_to_display_name = {}

        self.rebuild_index()

    def rebuild_index(self):
        """Rebuild FAISS index from backend database embeddings."""
        vectors, person_ids, id_to_display_name = load_all_embeddings(
            self.db, self.user_id, self.dim
        )
        self.id_to_display_name = id_to_display_name

        if vectors.shape[0] == 0:
            self.index = faiss.IndexFlatL2(self.dim)
            self.person_ids = np.empty((0,), dtype=object)
            return

        # Normalize vectors for stability (face models use cosine similarity).
        # If we normalize, L2 distance corresponds to cosine similarity.
        vectors = self._l2_normalize(vectors)

        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(vectors)
        self.person_ids = person_ids

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, eps)

    def enroll(self, frame: np.ndarray, first_name: str, last_name: str) -> bool:
        """Enroll a new person with face embedding from the current frame."""
        faces = self.app.get(frame)
        if not faces:
            return False

        # Pick the largest face
        def area(f):
            x1, y1, x2, y2 = f.bbox
            return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        
        face = max(faces, key=area)

        emb = face.embedding.astype(np.float32).reshape(1, -1)
        if emb.shape[1] != self.dim:
            self.dim = emb.shape[1]

        # Normalize the embedding
        emb = self._l2_normalize(emb)[0]

        # Create new Person in backend database
        person = Person(
            user_id=self.user_id,
            first_name=first_name,
            last_name=last_name,
            display_name=f"{first_name} {last_name}",
            face_embedding=emb.tolist(),
            face_embedding_model=MODEL_IDENTIFIER,
            last_seen_at=datetime.now(timezone.utc)
        )
        self.db.add(person)
        self.db.commit()
        self.db.refresh(person)
        
        self.rebuild_index()
        return True

    def recognize_faces(self, frame: np.ndarray, k: int = 1):
        """Recognize faces in the frame and return results."""
        faces = self.app.get(frame)
        results = []
        if not faces:
            return results

        for f in faces:
            bbox = f.bbox.astype(int)
            emb = f.embedding.astype(np.float32).reshape(1, -1)
            if emb.shape[1] != self.dim:
                # If model changes dimension (unlikely), rebuild everything
                self.dim = emb.shape[1]
                self.rebuild_index()

            emb = self._l2_normalize(emb)

            name = "Unknown"
            score = None
            dist = None
            person_id = None

            if self.index is not None and self.index.ntotal > 0:
                D, I = self.index.search(emb, k)
                dist = float(D[0][0])
                idx = int(I[0][0])
                if idx >= 0 and dist <= L2_DIST_THRESHOLD:
                    person_id = self.person_ids[idx]
                    name = self.id_to_display_name.get(person_id, "Unknown")
                    
                    # Update last_seen_at in database
                    person = self.db.query(Person).filter(Person.id == person_id).first()
                    if person:
                        person.last_seen_at = datetime.now(timezone.utc)
                        self.db.commit()

                # Convert normalized-L2 distance to cosine similarity:
                # if vectors normalized, L2^2 = 2 - 2*cos -> cos = 1 - (L2^2)/2
                score = 1.0 - (dist / 2.0)

            results.append({
                "bbox": bbox,
                "name": name,
                "dist": dist,
                "score": score,
                "person_id": str(person_id) if person_id else None
            })

        return results

def draw_overlay(frame: np.ndarray, recogs):
    out = frame.copy()
    for r in recogs:
        x1, y1, x2, y2 = r["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = r["name"]
        if r["dist"] is not None:
            label += f"  d={r['dist']:.2f}"
        cv2.putText(out, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out

def main():
    print("=" * 60)
    print("Face Recognition - Backend Database Integration")
    print("=" * 60)
    print()
    print("Controls:")
    print("  E = enroll new person (enter first and last name)")
    print("  R = rebuild index from database")
    print("  Q / ESC = quit")
    print()
    print(f"Database: {settings.database_url}")
    print(f"L2_DIST_THRESHOLD: {L2_DIST_THRESHOLD} (lower = stricter matching)")
    print(f"Embedding dimension: {settings.embedding_dimension}")
    print(f"Embedding model: {MODEL_IDENTIFIER}")
    print()

    # Connect to backend database
    db = SessionLocal()
    
    try:
        # Get user ID
        user_id = get_user_id(db)
        print(f"Using User ID: {user_id}")
        print()
        
        recog = FaceRecognizer(db, user_id)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam (index 0). Try changing VideoCapture(0) to (1).")
            sys.exit(1)

        last_prompt_time = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: failed to read frame")
                break

            recogs = recog.recognize_faces(frame, k=1)
            view = draw_overlay(frame, recogs)

            cv2.imshow("Face Recognition (Backend Database)", view)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q'), ord('Q')):
                break

            elif key in (ord('r'), ord('R')):
                recog.rebuild_index()
                print("Rebuilt FAISS index from backend database.")

            elif key in (ord('e'), ord('E')):
                # avoid repeated enroll triggers if key repeats
                if time.time() - last_prompt_time < 0.5:
                    continue
                last_prompt_time = time.time()

                first_name = input("Enter first name to enroll: ").strip()
                if not first_name:
                    print("Skipped (empty first name).")
                    continue
                
                last_name = input("Enter last name to enroll: ").strip()
                if not last_name:
                    print("Skipped (empty last name).")
                    continue

                success = recog.enroll(frame, first_name, last_name)
                if success:
                    print(f"Enrolled: {first_name} {last_name}")
                else:
                    print("No face detected to enroll.")

        cap.release()
        cv2.destroyAllWindows()

    finally:
        db.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()