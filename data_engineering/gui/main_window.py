
import os
import sys
from PyQt6.QtWidgets import (
    QMainWindow, QToolBar, QWidget, QVBoxLayout, QFileDialog, QInputDialog
)
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtCore import QSize

from canvas import Canvas

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Sequence Viewer')
        self.resize(800, 600)
        
        self.current_series_path = None

        # Toolbar and Actions
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Action for selecting a timeseries folder
        series_act = QAction("Select Series", self)
        series_act.setStatusTip("Select a folder containing TIFF images")
        series_act.triggered.connect(self.on_select_series)
        toolbar.addAction(series_act)
        
        toolbar.addSeparator()

        # Action for previous
        prev_act = QAction("Previous", self)
        prev_act.setStatusTip("Go to the previous image")
        prev_act.triggered.connect(self.on_previous)
        toolbar.addAction(prev_act)
        
        # Action for play
        self.play_act = QAction("Play", self)
        self.play_act.setStatusTip("Play the image sequence")
        self.play_act.triggered.connect(self.on_play)
        toolbar.addAction(self.play_act)

        # Action for stop
        self.stop_act = QAction("Stop", self)
        self.stop_act.setStatusTip("Stop the animation")
        self.stop_act.triggered.connect(self.on_stop)
        toolbar.addAction(self.stop_act)

        # Action for next
        next_act = QAction("Next", self)
        next_act.setStatusTip("Go to the next image")
        next_act.triggered.connect(self.on_next)
        toolbar.addAction(next_act)
        
        toolbar.addSeparator()
        
        # Action for frame rate 
        fps_act = QAction("Frame Rate", self)
        fps_act.setStatusTip("Adjust the animation speed (frames per second)")
        fps_act.triggered.connect(self.on_set_frame_rate)
        toolbar.addAction(fps_act)
        
        # Central widget and layout
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        
        # Create and add the canvas
        self.canvas = Canvas(self)
        layout.addWidget(self.canvas)

    def on_select_series(self):
        """Opens a dialog to select a directory and loads the image series."""
        start_path = os.getcwd()
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Image Series Folder", start_path
        )
        if folder_path:
            self.current_series_path = folder_path
            self.canvas.load_series(folder_path)

    def on_stop(self):
        self.canvas.stop()

    def on_play(self):
        if self.current_series_path:
            self.canvas.play()
        else:
            self.on_select_series()

    def on_next(self):
        self.canvas.next_frame()

    def on_previous(self):
        self.canvas.previous_frame()

    def on_set_frame_rate(self):
        """Opens a dialog to get a new FPS value from the user."""
        current_fps = self.canvas.get_frame_rate()
        new_fps, ok = QInputDialog.getInt(
            self,
            "Set Animation Speed",
            "Frames per second:",
            current_fps,  # Initial value
            1,            # Minimum value
            100,          # Maximum value
            1             # Step size
        )
        if ok:
            self.canvas.set_frame_rate(new_fps)