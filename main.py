import io
import os
from datetime import datetime
from typing import Any

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from database import DatabaseManager
from report_generator import generate_pdf_report

torch.classes.__path__ = []

MENU_STYLE = """<style>MainMenu {visibility: hidden;}</style>"""

TITLE_HTML = """<div><h1 style="color:#111F68; text-align:center; font-size:40px; margin-top:-50px;
font-family: 'Archivo', sans-serif; margin-bottom:20px;">MTUCI Shop Detector</h1></div>"""

SUBTITLE_HTML = """<div><h5 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif;
margin-top:-15px; margin-bottom:50px;">Real-time person detection system for webcam, video, and image analysis</h5></div>"""

LOGO_SVG = """
<svg width="250" height="100" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#111F68;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#042AFF;stop-opacity:1" />
        </linearGradient>
    </defs>
    <rect width="250" height="100" fill="url(#grad1)" rx="10"/>
    <path d="M 125 15 L 145 25 L 150 42 L 140 57 L 110 57 L 100 42 L 105 25 Z"
          fill="none" stroke="white" stroke-width="3.5"
          stroke-linejoin="round" stroke-linecap="round"
          style="filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.3));"/>
    <text x="125" y="85" font-family="Arial, sans-serif" font-size="16"
          font-weight="bold" fill="white" text-anchor="middle">
        MTUCI Shop Detector
    </text>
</svg>
"""

STATS_CARD_TEMPLATE = """
<div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 10px;">
    <h3 style="color: #111F68; margin-bottom: 15px;">Detection Statistics</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="color: #666; margin: 0; font-size: 14px;">Current</p>
            <p style="color: #042AFF; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;">{current}</p>
        </div>
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="color: #666; margin: 0; font-size: 14px;">Average</p>
            <p style="color: #28a745; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;">{average:.1f}</p>
        </div>
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="color: #666; margin: 0; font-size: 14px;">Minimum</p>
            <p style="color: #17a2b8; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;">{minimum}</p>
        </div>
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="color: #666; margin: 0; font-size: 14px;">Maximum</p>
            <p style="color: #dc3545; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;">{maximum}</p>
        </div>
    </div>
</div>
"""

BEST_MODELS = ["YOLO11n", "YOLO11s", "YOLO11m"]
DEFAULT_MODEL_INDEX = 1
TEMP_VIDEO_FILE = "ultralytics.mp4"


class Inference:
    """A class to perform object detection, image classification, image segmentation and pose estimation inference.

    This class provides functionalities for loading models, configuring settings, uploading video files, and performing
    real-time inference using Streamlit and MTUCI Shop Detector YOLO models.

    Attributes:
        st (module): Streamlit module for UI creation.
        temp_dict (dict): Temporary dictionary to store the model path and other configuration.
        model_path (str): Path to the loaded model.
        model (YOLO): The YOLO model instance.
        source (str): Selected video source (webcam or video file).
        enable_trk (bool): Enable tracking option.
        conf (float): Confidence threshold for detection.
        iou (float): IoU threshold for non-maximum suppression.
        org_frame (Any): Container for the original frame to be displayed.
        ann_frame (Any): Container for the annotated frame to be displayed.
        vid_file_name (str | int): Name of the uploaded video file or webcam index.
        selected_ind (list[int]): List of selected class indices for detection.

    Methods:
        web_ui: Set up the Streamlit web interface with custom HTML elements.
        sidebar: Configure the Streamlit sidebar for model and inference settings.
        source_upload: Handle video file uploads through the Streamlit interface.
        configure: Configure the model and load selected classes for inference.
        inference: Perform real-time object detection inference.

    Examples:
        Create an Inference instance with a custom model
        >>> inf = Inference(model="path/to/model.pt")
        >>> inf.inference()

        Create an Inference instance with default settings
        >>> inf = Inference()
        >>> inf.inference()
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Inference class, checking Streamlit requirements and setting up the model path.

        Args:
            **kwargs (Any): Additional keyword arguments for model configuration.
        """
        check_requirements("streamlit>=1.29.0")
        import streamlit as st

        self.st = st
        self.source = None
        self.img_file_names = []
        self.enable_trk = False
        self.conf = 0.5
        self.iou = 0.4
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.original_video_name = None
        self.selected_ind: list[int] = []
        self.model = None
        self.selected_model_name = None

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = self.temp_dict.get("model")

        self._initialize_session()
        print(f"Session ID: {self._get_session_id()}")

        # Initialize database in session_state to persist across reloads
        if "db" not in self.st.session_state:
            try:
                self.st.session_state.db = DatabaseManager()
                print(
                    f"✅ Database initialized successfully: connected={self.st.session_state.db.connected}"
                )
            except Exception as e:
                print(f"❌ Database initialization failed: {e}")
                self.st.session_state.db = None

        self.db = self.st.session_state.db
        print(
            f"📊 Using database from session: connected={self.db.connected if self.db else False}"
        )

        LOGGER.info(f"MTUCI Shop Detector Solutions: ✅ {self.temp_dict}")

    def _initialize_session(self) -> None:
        """Initialize session ID if not exists."""
        if "session_id" not in self.st.session_state:
            import uuid

            self.st.session_state["session_id"] = str(uuid.uuid4())

    def _get_session_id(self) -> str:
        """Get current session ID."""
        return self.st.session_state.get("session_id", "unknown")

    def web_ui(self) -> None:
        """Set up the Streamlit web interface with custom HTML elements."""
        self.st.set_page_config(
            page_title="MTUCI Shop Detector Streamlit App", layout="wide"
        )
        self.st.markdown(MENU_STYLE, unsafe_allow_html=True)
        self.st.markdown(TITLE_HTML, unsafe_allow_html=True)
        self.st.markdown(SUBTITLE_HTML, unsafe_allow_html=True)

    def sidebar(self) -> None:
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:
            self.st.markdown(LOGO_SVG, unsafe_allow_html=True)

        self.st.sidebar.title("User Configuration")
        self.source = self.st.sidebar.selectbox(
            "Source",
            ("webcam", "video", "image"),
        )
        if self.source in ["webcam", "video"]:
            self.enable_trk = (
                self.st.sidebar.radio("Enable Tracking", ("Yes", "No")) == "Yes"
            )
        self.conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.05)
        )
        self.iou = float(
            self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.05)
        )

        if self.st.sidebar.button("📊 Download Analytics Report"):
            self.generate_report()

        if self.source != "image":
            col1, col2 = self.st.columns(2)
            self.org_frame = col1.empty()
            self.ann_frame = col2.empty()
            self.person_counter = self.st.empty()

    def source_upload(self) -> None:
        """Handle video file uploads through the Streamlit interface."""
        from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

        self.vid_file_name = ""
        self.original_video_name = None
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader(
                "Upload Video File", type=VID_FORMATS
            )
            if vid_file is not None:
                self.original_video_name = vid_file.name
                video_bytes = io.BytesIO(vid_file.read())
                with open(TEMP_VIDEO_FILE, "wb") as out:
                    out.write(video_bytes.read())
                self.vid_file_name = TEMP_VIDEO_FILE
        elif self.source == "webcam":
            self.vid_file_name = 0
            self.original_video_name = "webcam"
        elif self.source == "image":
            import tempfile

            if imgfiles := self.st.sidebar.file_uploader(
                "Upload Image Files", type=IMG_FORMATS, accept_multiple_files=True
            ):
                for imgfile in imgfiles:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f".{imgfile.name.split('.')[-1]}"
                    ) as tf:
                        tf.write(imgfile.read())
                        self.img_file_names.append(
                            {"path": tf.name, "name": imgfile.name}
                        )

    def configure(self) -> None:
        """Configure the model and load selected classes for inference."""
        models = BEST_MODELS.copy()
        if self.model_path:
            models.insert(0, self.model_path)

        selected_model = self.st.sidebar.selectbox(
            "Detection Model", models, index=DEFAULT_MODEL_INDEX
        )

        with self.st.spinner("Model is downloading..."):
            if selected_model.endswith(
                (".pt", ".onnx", ".torchscript", ".mlpackage", ".engine")
            ) or any(fmt in selected_model for fmt in ("openvino_model", "rknn_model")):
                model_path = selected_model
            else:
                model_path = f"{selected_model.lower()}.pt"
            self.model = YOLO(model_path)
            self.selected_model_name = selected_model
            class_names = list(self.model.names.values())
        self.st.success("Model loaded successfully!")

        if "person" in class_names:
            self.selected_ind = [class_names.index("person")]
            self.st.sidebar.info("Detecting: Person")
        else:
            self.st.sidebar.warning(
                "Person class not found in model, using all available classes"
            )
            self.selected_ind = list(range(len(class_names)))

    def image_inference(self) -> None:
        """Perform inference on uploaded images."""
        for img_info in self.img_file_names:
            img_path = img_info["path"]
            image = cv2.imread(img_path)
            if image is not None:
                self.st.markdown(f"#### Processed: {img_info['name']}")
                col1, col2 = self.st.columns(2)
                with col1:
                    self.st.image(image, channels="BGR", caption="Original Image")
                results = self.model(
                    image, conf=self.conf, iou=self.iou, classes=self.selected_ind
                )
                annotated_image = results[0].plot()

                person_count = len(results[0].boxes)

                with col2:
                    self.st.image(
                        annotated_image, channels="BGR", caption="Predicted Image"
                    )

                self.st.markdown(f"### Detected Persons: **{person_count}**")

                self._save_image_analytics(img_info["name"], person_count)

                try:
                    os.unlink(img_path)
                except FileNotFoundError:
                    pass
            else:
                self.st.error("Could not load the uploaded image.")

    def _save_image_analytics(self, file_name: str, person_count: int) -> None:
        """Save image analytics to database."""
        if not self.db:
            print("Database not initialized")
            return
        if not self.db.connected:
            print("Database not connected")
            return

        print(
            f"Saving image analytics: file={file_name}, count={person_count}, session={self._get_session_id()}"
        )
        self.db.save_image_analytics(
            session_id=self._get_session_id(),
            file_name=file_name,
            person_count=person_count,
            confidence=self.conf,
            iou=self.iou,
            model_name=self.selected_model_name or "unknown",
        )
        print("Image analytics saved successfully")

    def _save_video_analytics(self, file_name: str, person_counts: list[int]) -> None:
        """Save video analytics to database."""
        print(f"🎯 _save_video_analytics CALLED!")
        print(f"   file_name: {file_name}")
        print(f"   person_counts: {person_counts[:5] if person_counts else 'EMPTY'}")
        print(f"   len(person_counts): {len(person_counts) if person_counts else 0}")
        print(f"   self.db: {self.db}")
        print(f"   self.db.connected: {self.db.connected if self.db else 'N/A'}")

        if not self.db:
            print("❌ Database not initialized")
            return
        if not self.db.connected:
            print("❌ Database not connected")
            return
        if not person_counts:
            print("❌ No person counts to save")
            return

        print(
            f"✅ All checks passed! Calling db.save_video_analytics..."
        )
        print(
            f"   Params: session={self._get_session_id()}, file={file_name}, counts={len(person_counts)}"
        )
        self.db.save_video_analytics(
            session_id=self._get_session_id(),
            file_name=file_name,
            person_counts=person_counts,
            confidence=self.conf,
            iou=self.iou,
            model_name=self.selected_model_name or "unknown",
        )
        print("✅ Video analytics saved successfully")

    def generate_report(self) -> None:
        """Generate and download PDF report for current session."""
        if not self.db or not self.db.connected:
            self.st.error("Database not connected. Cannot generate report.")
            return

        with self.st.spinner("Generating report..."):
            session_data = self.db.get_session_analytics(self._get_session_id())

            if not session_data:
                self.st.warning(
                    "No analytics data found for your session. Process some images or videos first."
                )
                return

            try:
                pdf_buffer = generate_pdf_report(session_data, self._get_session_id())
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                self.st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"mtuci_analytics_{timestamp}.pdf",
                    mime="application/pdf",
                )
                self.st.success(
                    f"Report generated successfully. Found {len(session_data)} records."
                )
            except Exception as e:
                self.st.error(f"Error generating report: {e}")

    def inference(self) -> None:
        """Perform real-time object detection inference on video or webcam feed."""
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()

        if self.st.sidebar.button("Start"):
            if self.source == "image":
                if self.img_file_names:
                    self.image_inference()
                else:
                    self.st.info("Please upload an image file.")
                return

            stop_button = self.st.sidebar.button("Stop")
            cap = cv2.VideoCapture(self.vid_file_name)
            if not cap.isOpened():
                self.st.error("Could not open webcam or video source.")
                return

            # Initialize person_counts in session_state if not exists
            if "person_counts" not in self.st.session_state:
                self.st.session_state.person_counts = []

            person_counts = self.st.session_state.person_counts

            try:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        if self.source == "video":
                            self.st.info("Video processing completed.")
                        else:
                            self.st.warning(
                                "Failed to read frame from webcam. Please verify the webcam is connected properly."
                            )
                        break

                    if self.enable_trk:
                        results = self.model.track(
                            frame,
                            conf=self.conf,
                            iou=self.iou,
                            classes=self.selected_ind,
                            persist=True,
                        )
                    else:
                        results = self.model(
                            frame,
                            conf=self.conf,
                            iou=self.iou,
                            classes=self.selected_ind,
                        )

                    annotated_frame = results[0].plot()

                    person_count = len(results[0].boxes)
                    person_counts.append(person_count)
                    self.st.session_state.person_counts = (
                        person_counts  # Save to session
                    )

                    if stop_button:
                        print(f"🛑 Stop button pressed, saving analytics...")
                        file_name = self._get_video_file_name()
                        self._save_video_analytics(file_name, person_counts)
                        self.st.session_state.person_counts = []  # Clear after saving
                        break

                    self.org_frame.image(
                        frame, channels="BGR", caption="Original Frame"
                    )
                    self.ann_frame.image(
                        annotated_frame, channels="BGR", caption="Predicted Frame"
                    )

                    self._display_statistics(person_count, person_counts)
            finally:
                print(
                    f"🔄 Finally block executing... person_counts length: {len(person_counts)}"
                )
                cap.release()
                # Only save if we have data and haven't saved yet (video ended naturally)
                if person_counts and not stop_button:
                    file_name = self._get_video_file_name()
                    print(
                        f"📝 Video ended naturally, saving analytics for: {file_name}"
                    )
                    self._save_video_analytics(file_name, person_counts)
                    self.st.session_state.person_counts = []  # Clear after saving
                    print(f"✅ Analytics saved, stopping Streamlit execution")
                    import time
                    time.sleep(0.5)  # Give time for DB commit
                print(f"✅ Finally block completed")

        cv2.destroyAllWindows()

    def _get_video_file_name(self) -> str:
        """Get video file name for analytics."""
        if self.original_video_name:
            return self.original_video_name
        return "webcam" if self.source == "webcam" else str(self.vid_file_name)

    def _display_statistics(self, current_count: int, person_counts: list[int]) -> None:
        """Display detection statistics."""
        if person_counts:
            stats_html = STATS_CARD_TEMPLATE.format(
                current=current_count,
                average=sum(person_counts) / len(person_counts),
                minimum=min(person_counts),
                maximum=max(person_counts),
            )
            self.person_counter.markdown(stats_html, unsafe_allow_html=True)
        else:
            self.person_counter.markdown(f"### Detected Persons: **{current_count}**")


def main() -> None:
    """Main entry point for the application."""
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else None
    Inference(model=model).inference()


if __name__ == "__main__":
    main()
