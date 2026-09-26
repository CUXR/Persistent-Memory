"""Local face recognition: unseen faces receive persistent person IDs automatically."""
import argparse
import os
from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.person import Person
from app.models.user import User

L2_DIST_THRESHOLD = 1.0
MIN_DETECTION_SCORE = 0.7
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
MODEL_IDENTIFIER = f"{MODEL_NAME}_insightface"


def load_all_embeddings(db: Session, user_id: UUID, dim: int):
    """Only compare this owner's faces encoded by the same model."""
    people = db.scalars(select(Person).where(
        Person.user_id == user_id, Person.face_embedding_model == MODEL_IDENTIFIER,
    )).all()
    vectors, person_ids, names = [], [], {}
    for person in people:
        if person.face_embedding is None:
            continue
        vector = np.asarray(person.face_embedding, dtype=np.float32)
        if vector.shape != (dim,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-10:
            continue
        vectors.append(vector)
        person_ids.append(person.id)
        names[person.id] = person.display_name or f"{person.first_name} {person.last_name}".strip()
    return np.asarray(vectors, dtype=np.float32).reshape(-1, dim), person_ids, names


class FaceRecognizer:
    def __init__(self, db: Session, user_id: UUID, *, detector=None, index_factory=None):
        self.db = db
        self.user_id = user_id
        if db.get(User, user_id) is None:
            raise ValueError("Face recognition owner not found")
        if detector is None:
            from insightface.app import FaceAnalysis
            detector = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
            detector.prepare(ctx_id=0, det_size=DET_SIZE)
        if index_factory is None:
            import faiss
            index_factory = faiss.IndexFlatL2
        self.app = detector
        self._index_factory = index_factory
        self.dim = get_settings().embedding_dimension
        self.rebuild_index()

    @staticmethod
    def _l2_normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms < 1e-10):
            raise ValueError("Face embedding must be finite and nonzero")
        return vectors / norms

    def rebuild_index(self):
        vectors, self.person_ids, self.id_to_display_name = load_all_embeddings(
            self.db, self.user_id, self.dim,
        )
        self.index = self._index_factory(self.dim)
        if len(vectors):
            self.index.add(self._l2_normalize(vectors))

    def _create_unknown_person(self, embedding):
        person_id = uuid4()
        person = Person(
            id=person_id, user_id=self.user_id, first_name="Unknown", last_name="",
            display_name=f"Unknown person {person_id}",
            face_embedding=embedding.tolist(), face_embedding_model=MODEL_IDENTIFIER,
            last_seen_at=datetime.now(timezone.utc),
        )
        try:
            self.db.add(person)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        # Update before the next face/frame, so the same face reuses this ID.
        self.rebuild_index()
        return person

    def recognize_faces(self, frame: np.ndarray):
        """Return persisted person IDs for known and newly encountered faces.

        Creation needs no name or user prompt. The UUID remains stable if the
        person is named later. Several faces return separate IDs; selecting the
        active interlocutor is the tracker's responsibility.
        """
        results = []
        for face in self.app.get(frame):
            if face.det_score < MIN_DETECTION_SCORE:
                continue
            embedding = np.asarray(face.embedding, dtype=np.float32).reshape(1, -1)
            if embedding.shape[1] != self.dim:
                raise ValueError("Face model dimension differs from the database configuration")
            embedding = self._l2_normalize(embedding)
            person, distance = None, None
            if self.index.ntotal:
                distances, indices = self.index.search(embedding, 1)
                distance, index = float(distances[0][0]), int(indices[0][0])
                if index >= 0 and distance <= L2_DIST_THRESHOLD:
                    candidate = self.db.get(Person, self.person_ids[index])
                    if candidate is not None and candidate.user_id == self.user_id:
                        person = candidate
            created = person is None
            if created:
                person = self._create_unknown_person(embedding[0])
                # A newly created identity has no independent face-match score.
                distance = None
            else:
                person.last_seen_at = datetime.now(timezone.utc)
                self.db.commit()
            results.append({
                "bbox": np.asarray(face.bbox, dtype=int),
                "name": person.display_name or f"{person.first_name} {person.last_name}".strip(),
                "dist": distance,
                "score": None if distance is None else float(np.clip(1 - distance / 2, -1, 1)),
                "person_id": str(person.id),
                "created": created,
            })
        return results

    def get_current_person_id(self, frame: np.ndarray) -> UUID | None:
        """Provide an audio target only when exactly one face is visible."""
        faces = self.recognize_faces(frame)
        return UUID(faces[0]["person_id"]) if len(faces) == 1 else None


def draw_overlay(frame, recognitions):
    import cv2
    output = frame.copy()
    for recognition in recognitions:
        x1, y1, x2, y2 = recognition["bbox"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = recognition["name"]
        if label.startswith("Unknown person "):
            label = f"Unknown ({recognition['person_id'][:8]})"
        cv2.putText(output, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-id", type=UUID, default=os.getenv("USER_ID"))
    args = parser.parse_args()
    if args.owner_id is None:
        parser.error("Provide --owner-id or USER_ID for an existing user")
    import cv2
    print("Unknown faces are saved automatically. R = rebuild index; Q / ESC = quit.")
    with SessionLocal() as db:
        recognizer = FaceRecognizer(db, args.owner_id)
        camera = cv2.VideoCapture(0)
        try:
            if not camera.isOpened():
                raise RuntimeError("Could not open webcam")
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                faces = recognizer.recognize_faces(frame)
                for face in faces:
                    if face["created"]:
                        print(f"Created person_id={face['person_id']}")
                cv2.imshow("Persistent Memory", draw_overlay(frame, faces))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    recognizer.rebuild_index()
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
