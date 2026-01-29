# MTUCI Shop Detector 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import os
from typing import Any

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

from database import DatabaseManager

torch.classes.__path__ = []  # Torch module __path__._path issue: https://github.com/datalab-to/marker/issues/442


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
        check_requirements(
            "streamlit>=1.29.0"
        )  # scope imports for faster ultralytics package load speeds
        import streamlit as st

        self.st = st  # Reference to the Streamlit module
        self.source = None  # Video source selection (webcam or video file)
        self.img_file_names = []  # List of image file names
        self.enable_trk = False  # Flag to toggle object tracking
        self.conf = 0.5  # Confidence threshold for detection (higher for better person detection)
        self.iou = (
            0.4  # Intersection-over-Union (IoU) threshold for non-maximum suppression
        )
        self.org_frame = None  # Container for the original frame display
        self.ann_frame = None  # Container for the annotated frame display
        self.vid_file_name = None  # Video file name or webcam index
        self.selected_ind: list[
            int
        ] = []  # List of selected class indices for detection
        self.model = None  # YOLO model instance
        self.selected_model_name = None  # Selected model name for analytics

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None  # Model file path
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        # Initialize database manager
        try:
            self.db = DatabaseManager()
        except Exception as e:
            print(f"Database initialization failed: {e}")
            self.db = None

        # Initialize session_id if not exists
        if 'session_id' not in self.st.session_state:
            import uuid
            self.st.session_state['session_id'] = str(uuid.uuid4())

        LOGGER.info(f"MTUCI Shop Detector Solutions: ✅ {self.temp_dict}")

    def web_ui(self) -> None:
        """Set up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = (
            """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style
        )

        # Main title of streamlit application
        main_title_cfg = """<div><h1 style="color:#111F68; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">MTUCI Shop Detector Streamlit Application</h1></div>"""

        # Subtitle of streamlit application
        sub_title_cfg = """<div><h5 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif;
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam, videos, and images
        with the power of MTUCI Shop Detector! 🚀</h5></div>"""

        # Set html page configuration and append custom HTML
        self.st.set_page_config(
            page_title="MTUCI Shop Detector Streamlit App", layout="wide"
        )
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self) -> None:
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:  # Add MTUCI Shop Detector LOGO
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/MTUCI Shop Detector_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title(
            "User Configuration"
        )  # Add elements to vertical setting menu
        self.source = self.st.sidebar.selectbox(
            "Source",
            ("webcam", "video", "image"),
        )  # Add source selection dropdown
        if self.source in ["webcam", "video"]:
            self.enable_trk = (
                self.st.sidebar.radio("Enable Tracking", ("Yes", "No")) == "Yes"
            )  # Enable object tracking
        self.conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.05)
        )  # Slider for confidence
        self.iou = float(
            self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.05)
        )  # Slider for NMS threshold

        if self.source != "image":  # Only create columns for video/webcam
            col1, col2 = self.st.columns(2)  # Create two columns for displaying frames
            self.org_frame = col1.empty()  # Container for original frame
            self.ann_frame = col2.empty()  # Container for annotated frame
            self.person_counter = self.st.empty()  # Container for person counter

    def source_upload(self) -> None:
        """Handle video file uploads through the Streamlit interface."""
        from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS  # scope import

        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader(
                "Upload Video File", type=VID_FORMATS
            )
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # BytesIO Object
                with open(
                    "ultralytics.mp4", "wb"
                ) as out:  # Open temporary file as bytes
                    out.write(g.read())  # Read bytes into file
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0  # Use webcam index 0
        elif self.source == "image":
            import tempfile  # scope import

            if imgfiles := self.st.sidebar.file_uploader(
                "Upload Image Files", type=IMG_FORMATS, accept_multiple_files=True
            ):
                for imgfile in imgfiles:  # Save each uploaded image to a temporary file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f".{imgfile.name.split('.')[-1]}"
                    ) as tf:
                        tf.write(imgfile.read())
                        self.img_file_names.append(
                            {"path": tf.name, "name": imgfile.name}
                        )

    def configure(self) -> None:
        """Configure the model and load selected classes for inference."""
        # Best models for person detection (balanced quality/size)
        best_models = ["YOLO11n", "YOLO11s", "YOLO11m"]

        if self.model_path:  # Insert user provided custom model if available
            best_models.insert(0, self.model_path)

        selected_model = self.st.sidebar.selectbox(
            "Model (optimized for person detection)",
            best_models,
            index=1  # Default to YOLO11s (best balance)
        )

        with self.st.spinner("Model is downloading..."):
            if selected_model.endswith(
                (".pt", ".onnx", ".torchscript", ".mlpackage", ".engine")
            ) or any(fmt in selected_model for fmt in ("openvino_model", "rknn_model")):
                model_path = selected_model
            else:
                model_path = f"{selected_model.lower()}.pt"  # Default to .pt if no model provided during function call.
            self.model = YOLO(model_path)  # Load the YOLO model
            self.selected_model_name = selected_model  # Store model name for analytics
            class_names = list(
                self.model.names.values()
            )  # Convert dictionary to list of class names
        self.st.success("Model loaded successfully!")

        # Set to detect only 'person' class
        if "person" in class_names:
            self.selected_ind = [class_names.index("person")]
            self.st.sidebar.info("Detection class: person")
        else:
            self.st.sidebar.warning("'person' class not found in model, using all classes")
            self.selected_ind = list(range(len(class_names)))

    def image_inference(self) -> None:
        """Perform inference on uploaded images."""
        for img_info in self.img_file_names:
            img_path = img_info["path"]
            image = cv2.imread(img_path)  # Load and display the original image
            if image is not None:
                self.st.markdown(f"#### Processed: {img_info['name']}")
                col1, col2 = self.st.columns(2)
                with col1:
                    self.st.image(image, channels="BGR", caption="Original Image")
                results = self.model(
                    image, conf=self.conf, iou=self.iou, classes=self.selected_ind
                )
                annotated_image = results[0].plot()

                # Count detected persons
                person_count = len(results[0].boxes)

                with col2:
                    self.st.image(
                        annotated_image, channels="BGR", caption="Predicted Image"
                    )

                # Display person counter
                self.st.markdown(f"### 👥 Detected Persons: **{person_count}**")

                # Save analytics to database
                if self.db and self.db.connected:
                    session_id = self.st.session_state.get('session_id', 'unknown')
                    self.db.save_image_analytics(
                        session_id=str(session_id),
                        file_name=img_info['name'],
                        person_count=person_count,
                        confidence=self.conf,
                        iou=self.iou,
                        model_name=self.selected_model_name or 'unknown'
                    )

                try:  # Clean up temporary file
                    os.unlink(img_path)
                except FileNotFoundError:
                    pass  # File doesn't exist, ignore
            else:
                self.st.error("Could not load the uploaded image.")

    def inference(self) -> None:
        """Perform real-time object detection inference on video or webcam feed."""
        self.web_ui()  # Initialize the web interface
        self.sidebar()  # Create the sidebar
        self.source_upload()  # Upload the video source
        self.configure()  # Configure the app

        if self.st.sidebar.button("Start"):
            if self.source == "image":
                if self.img_file_names:
                    self.image_inference()
                else:
                    self.st.info("Please upload an image file to perform inference.")
                return

            stop_button = self.st.sidebar.button("Stop")  # Button to stop the inference
            cap = cv2.VideoCapture(self.vid_file_name)  # Capture the video
            if not cap.isOpened():
                self.st.error("Could not open webcam or video source.")
                return

            person_counts = []  # Track person counts for video analytics

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    if self.source == "video":
                        self.st.info("✅ Video processing completed!")
                    else:
                        self.st.warning(
                            "Failed to read frame from webcam. Please verify the webcam is connected properly."
                        )
                    break

                # Process frame with model
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
                        frame, conf=self.conf, iou=self.iou, classes=self.selected_ind
                    )

                annotated_frame = results[0].plot()  # Add annotations on frame

                # Count detected persons
                person_count = len(results[0].boxes)
                person_counts.append(person_count)  # Track for analytics

                if stop_button:
                    cap.release()  # Release the capture

                    # Save video analytics before stopping
                    if self.db and self.db.connected and person_counts:
                        session_id = self.st.session_state.get('session_id', 'unknown')
                        file_name = str(self.vid_file_name) if self.source == "video" else "webcam"
                        self.db.save_video_analytics(
                            session_id=str(session_id),
                            file_name=file_name,
                            person_counts=person_counts,
                            confidence=self.conf,
                            iou=self.iou,
                            model_name=self.selected_model_name or 'unknown'
                        )

                    self.st.stop()  # Stop streamlit app

                self.org_frame.image(
                    frame, channels="BGR", caption="Original Frame"
                )  # Display original frame
                self.ann_frame.image(
                    annotated_frame, channels="BGR", caption="Predicted Frame"
                )  # Display processed

                # Display person counter
                self.person_counter.markdown(f"### 👥 Detected Persons: **{person_count}**")

            cap.release()  # Release the capture

            # Save video analytics after video ends
            if self.db and self.db.connected and person_counts:
                session_id = self.st.session_state.get('session_id', 'unknown')
                file_name = str(self.vid_file_name) if self.source == "video" else "webcam"
                self.db.save_video_analytics(
                    session_id=str(session_id),
                    file_name=file_name,
                    person_counts=person_counts,
                    confidence=self.conf,
                    iou=self.iou,
                    model_name=self.selected_model_name or 'unknown'
                )

        cv2.destroyAllWindows()  # Destroy all OpenCV windows


if __name__ == "__main__":
    import sys  # Import the sys module for accessing command-line arguments

    # Check if a model name is provided as a command-line argument
    args = len(sys.argv)
    model = (
        sys.argv[1] if args > 1 else None
    )  # Assign first argument as the model name if provided
    # Create an instance of the Inference class and run inference
    Inference(model=model).inference()
