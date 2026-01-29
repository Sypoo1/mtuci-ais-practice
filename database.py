import os
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

Base = declarative_base()


class DetectionAnalytics(Base):
    """Model for detection analytics."""

    __tablename__ = "detection_analytics"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(255))
    file_name = Column(String(500))
    file_type = Column(String(50))
    person_count = Column(Integer)
    person_count_min = Column(Integer)
    person_count_max = Column(Integer)
    person_count_avg = Column(Float)
    confidence_threshold = Column(Float)
    iou_threshold = Column(Float)
    model_name = Column(String(100))


class DatabaseManager:
    """Manager for PostgreSQL database operations."""

    def __init__(self):
        """Initialize database connection."""
        self.session: Optional[Session] = None
        self.connected = False

        try:
            db_url = self._build_connection_url()
            self.engine = create_engine(db_url)
            Base.metadata.create_all(self.engine)
            SessionLocal = sessionmaker(bind=self.engine)
            self.session = SessionLocal()
            self.connected = True
        except Exception as e:
            print(f"Database connection error: {e}")

    def _build_connection_url(self) -> str:
        """Build database connection URL from environment variables."""
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "mtuci_shop_detector")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "postgres")
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def save_image_analytics(
        self,
        session_id: str,
        file_name: str,
        person_count: int,
        confidence: float,
        iou: float,
        model_name: str,
    ) -> None:
        """Save analytics for image processing."""
        if not self.connected or not self.session:
            return

        try:
            record = DetectionAnalytics(
                session_id=session_id,
                file_name=file_name,
                file_type="image",
                person_count=person_count,
                confidence_threshold=confidence,
                iou_threshold=iou,
                model_name=model_name,
            )
            self.session.add(record)
            self.session.commit()
        except Exception as e:
            print(f"Error saving image analytics: {e}")
            if self.session:
                self.session.rollback()

    def save_video_analytics(
        self,
        session_id: str,
        file_name: str,
        person_counts: list[int],
        confidence: float,
        iou: float,
        model_name: str,
    ) -> None:
        """Save analytics for video processing."""
        if not self.connected or not self.session or not person_counts:
            return

        try:
            record = DetectionAnalytics(
                session_id=session_id,
                file_name=file_name,
                file_type="video",
                person_count_min=min(person_counts),
                person_count_max=max(person_counts),
                person_count_avg=sum(person_counts) / len(person_counts),
                confidence_threshold=confidence,
                iou_threshold=iou,
                model_name=model_name,
            )
            self.session.add(record)
            self.session.commit()
        except Exception as e:
            print(f"Error saving video analytics: {e}")
            if self.session:
                self.session.rollback()

    def get_session_analytics(self, session_id: str) -> list[dict]:
        """Get all analytics for a specific session."""
        if not self.connected or not self.session:
            return []

        try:
            records = (
                self.session.query(DetectionAnalytics)
                .filter(DetectionAnalytics.session_id == session_id)
                .order_by(DetectionAnalytics.id.desc())
                .all()
            )

            return [self._record_to_dict(record) for record in records]
        except Exception as e:
            print(f"Error getting session analytics: {e}")
            return []

    def _record_to_dict(self, record: DetectionAnalytics) -> dict:
        """Convert database record to dictionary."""
        return {
            "id": record.id,
            "file_name": record.file_name,
            "file_type": record.file_type,
            "person_count": record.person_count,
            "person_count_min": record.person_count_min,
            "person_count_max": record.person_count_max,
            "person_count_avg": record.person_count_avg,
            "confidence_threshold": record.confidence_threshold,
            "iou_threshold": record.iou_threshold,
            "model_name": record.model_name,
        }

    def close(self) -> None:
        """Close database session."""
        if self.session:
            self.session.close()
