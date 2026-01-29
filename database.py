import os
from typing import List

from dotenv import load_dotenv
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "mtuci_shop_detector")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "postgres")

        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        try:
            self.engine = create_engine(db_url)
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            self.connected = True
        except Exception as e:
            print(f"Database connection error: {e}")
            self.connected = False
            self.session = None

    def save_image_analytics(
        self,
        session_id: str,
        file_name: str,
        person_count: int,
        confidence: float,
        iou: float,
        model_name: str,
    ):
        """Save analytics for image processing."""
        if not self.connected:
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
            self.session.rollback()

    def save_video_analytics(
        self,
        session_id: str,
        file_name: str,
        person_counts: List[int],
        confidence: float,
        iou: float,
        model_name: str,
    ):
        """Save analytics for video/webcam processing."""
        if not self.connected or not person_counts:
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
            self.session.rollback()

    def close(self):
        """Close database session."""
        if self.session:
            self.session.close()
