import os
import sys
from PyQt6.QtWidgets import QMainWindow, QToolBar, QWidget, QVBoxLayout, QFileDialog
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtCore import QSize

from canvas import Canvas

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Sequence Viewer')
        self.resize(800, 600)
        
        self.current_series_path = None

        # -- Toolbar and Actions --
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Action for selecting a timeseries folder
        series_act = QAction("Select Series", self)
        series_act.setStatusTip("Select a folder containing TIFF images")
        series_act.triggered.connect(self.on_select_series)
        toolbar.addAction(series_act)
        
        toolbar.addSeparator()

        # Action for 'Previous'
        prev_act = QAction("Previous", self)
        prev_act.setStatusTip("Go to the previous image")
        prev_act.triggered.connect(self.on_previous)
        toolbar.addAction(prev_act)
        
        # Action for 'Play'
        self.play_act = QAction("Play", self)
        self.play_act.setStatusTip("Play the image sequence")
        self.play_act.triggered.connect(self.on_play)
        toolbar.addAction(self.play_act)

        # Action for 'Stop'
        self.stop_act = QAction("Stop", self)
        self.stop_act.setStatusTip("Stop the animation")
        self.stop_act.triggered.connect(self.on_stop)
        toolbar.addAction(self.stop_act)

        # Action for 'Next'
        next_act = QAction("Next", self)
        next_act.setStatusTip("Go to the next image")
        next_act.triggered.connect(self.on_next)
        toolbar.addAction(next_act)
        
        # Central widget and layout
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        
        # Create and add the canvas
        self.canvas = Canvas(self)
        layout.addWidget(self.canvas)

    def on_select_series(self):
        """Opens a dialog to select a directory and loads the image series."""
        # Use the current directory as a starting point, or the user's home directory
        start_path = os.getcwd()
        
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Image Series Folder",
            start_path
        )
        
        if folder_path:
            self.current_series_path = folder_path
            self.canvas.load_series(folder_path)

    def on_stop(self):
        """Stops the animation."""
        self.canvas.stop()

    def on_play(self):
        """Plays the animation if a series is loaded."""
        if self.current_series_path:
            self.canvas.play()
        else:
            # Optionally, prompt the user to select a series first
            self.on_select_series()

    def on_next(self):
        """Shows the next frame."""
        self.canvas.next_frame()

    def on_previous(self):
        """Shows the previous frame."""
        self.canvas.previous_frame()

