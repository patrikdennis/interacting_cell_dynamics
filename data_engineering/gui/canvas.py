import os
import re
import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QMessageBox
from PyQt6.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)

class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure, self.ax = plt.subplots()
        self.ax.axis('off')

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # -- State Variables --
        self.folder_path = None
        self.image_files = []
        self.current_index = -1
        self.is_playing = False
        self.fps = 10  # Default frames per second

        # hold plotlib image artist object
        self.image_artist = None

        # animation timer
        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / self.fps))
        self.timer.timeout.connect(self.next_frame)

        # layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def load_series(self, folder_path: str):
        """Loads, sorts, and displays the first image from a folder."""
        self.stop()
        # Reset the image artist when loading a new series
        self.image_artist = None
        
        self.folder_path = folder_path
        try:
            self.image_files = self._sort_files(folder_path)
            if not self.image_files:
                QMessageBox.warning(self, "No Images Found", "The selected folder contains no .tif files.")
                # Clear the axes if no images are found
                if self.image_artist:
                    self.image_artist.set_data([[]])
                self.ax.clear()
                self.ax.axis('off')
                self.canvas.draw()
                return
            
            self.current_index = 0
            self._show_current_frame()
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Files", f"Could not sort or load files.\nCheck filename format.\n\nError: {e}")

    def _show_current_frame(self):
        """Displays the image at the current index, preserving zoom."""
        if not self.image_files or self.current_index < 0:
            return

        file_path = os.path.join(self.folder_path, self.image_files[self.current_index])
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Warning: Could not read image file: {file_path}")
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.image_artist is None:
            # If first frame --> create the image artist
            self.image_artist = self.ax.imshow(img_rgb)
        else:
            # For subsequent frames update the image data
            self.image_artist.set_data(img_rgb)
        
        # The axis limits (zoom) preserved
        self.ax.axis('off')
        self.figure.tight_layout(pad=0)
        self.canvas.draw()

    def play(self):
        if self.image_files and not self.is_playing:
            self.is_playing = True
            self.timer.start()

    def stop(self):
        self.is_playing = False
        self.timer.stop()
        
    def set_frame_rate(self, fps: int):
        """Updates the animation speed."""
        self.fps = fps
        self.timer.setInterval(int(1000 / self.fps))

    def get_frame_rate(self) -> int:
        """Returns the current frames per second."""
        return self.fps

    def next_frame(self):
        if not self.image_files:
            return
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self._show_current_frame()

    def previous_frame(self):
        if not self.image_files:
            return
        self.current_index = (self.current_index - 1 + len(self.image_files)) % len(self.image_files)
        self._show_current_frame()

    def _sort_files(self, path_to_folder: str) -> list:
        dict_of_files = {}
        for entry in os.scandir(path_to_folder):
            if entry.is_file() and entry.name.lower().endswith('.tif'):
                try:
                    reduced_timestamp = self._reduce_timestamp(entry.name)
                    dict_of_files[entry.name] = int(reduced_timestamp)
                except (AttributeError, IndexError):
                    print(f"Skipping file with incorrect format: {entry.name}")
        sorted_items = sorted(dict_of_files.items(), key=lambda item: item[1])
        return [file_name for file_name, timestamp in sorted_items]

    def _reduce_timestamp(self, file_name: str) -> str:
        parts = file_name.split("_")
        timestamp_part = parts[3].split(".")[0]
        match = re.match(r"(\d+)d(\d+)h(\d+)m", timestamp_part)
        days, hours, minutes = match.groups()
        return f"{int(days):d}{int(hours):02d}{int(minutes):02d}"