import sys
import csv
from pathlib import Path

from PIL import Image
import re

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPen, QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QMessageBox, QToolBar, QSlider, QMenu
)



BASE_MARKER_RADIUS = 6.0

def pil_to_qpixmap(pil_image):
    """
    Convert a PIL Image to QPixmap.
    Here we convert everything to grayscale for simplicity.
    Adapt if you want RGB.
    """
    pil_image = pil_image.convert("L")  # grayscale
    w, h = pil_image.size
    data = pil_image.tobytes("raw", "L")
    qimage = QImage(data, w, h, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimage)

TIME_PATTERN = re.compile(r"(\d{2})d(\d{2})h(\d{2})m")


def time_key_from_path(path: Path):
    """
    Extract (days, hours, minutes) from a filename like:
    VID558_A1_1_00d00h00m.tif

    Returns a tuple (d, h, m). If the pattern is not found,
    returns (999, 999, 999) so those files appear last.
    """
    stem = path.stem  # filename without extension
    match = TIME_PATTERN.search(stem)
    if match:
        days = int(match.group(1))
        hours = int(match.group(2))
        minutes = int(match.group(3))
        return (days, hours, minutes)
    else:
        return (999, 999, 999)
class ImageView(QGraphicsView):
    """
    QGraphicsView subclass that shows an image and lets the user
    click to place markers. Supports zoom (via slider), panning with
    left mouse drag, and right-click context menu to delete a marker.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.image_item = None

        # list of dicts: {"item": QGraphicsEllipseItem, "x": float, "y": float, "r": float}
        self.markers = []

        # external callbacks
        self.click_callback = None      # function(x, y, r) when marker is added
        self.delete_callback = None     # function(x, y, r) when marker is deleted

        # zoom state
        self.base_zoom = 1.0           # scale that fits image to view
        self.current_zoom = 1.0        # absolute scale currently applied
        self.current_factor = 1.0      # relative factor (1.0 = base)
        self.min_factor = 0.1          # 10% of base
        self.max_factor = 10.0         # 1000% of base

        # marker radius in image coordinates at factor = 1.0
        self.base_marker_radius = BASE_MARKER_RADIUS

        # mouse state for distinguishing click vs drag (left button)
        self.left_mouse_pressed = False
        self.moved_since_press = False
        self.press_pos = None
        self.click_threshold = 5  # pixels in view coordinates

        # view settings
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        # tolerance for picking a marker on right-click (in image/scene coords)
        self.pick_radius = 10.0

    # ---------- Image & markers ----------

    def set_image(self, pixmap: QPixmap):
        """
        Display a new image, clearing old markers and resetting zoom.
        Automatically fits image to the view and sets base_zoom.
        """
        self.scene().clear()
        self.markers = []
        self.image_item = self.scene().addPixmap(pixmap)
        self.setSceneRect(self.image_item.boundingRect())

        # reset transform
        self.resetTransform()

        # compute scale that fits image into view
        view_rect = self.viewport().rect()
        scene_rect = self.image_item.boundingRect()

        if scene_rect.width() > 0 and scene_rect.height() > 0:
            sx = view_rect.width() / scene_rect.width()
            sy = view_rect.height() / scene_rect.height()
            scale = min(sx, sy)
        else:
            scale = 1.0

        # apply base zoom
        self.scale(scale, scale)
        self.base_zoom = scale
        self.current_factor = 1.0
        self.current_zoom = self.base_zoom * self.current_factor

    def clear_markers(self):
        for m in self.markers:
            self.scene().removeItem(m["item"])
        self.markers = []

    def _compute_radius_for_current_zoom(self) -> float:
        """Radius in image coords given current relative zoom factor."""
        factor = self.current_factor if self.current_factor > 0 else 1.0
        return self.base_marker_radius / factor

    def add_marker(self, x, y, r=None):
        """
        Add a circle marker at (x, y) in scene coordinates.

        If r is None, radius is computed from the current zoom.
        If r is provided, it is used as-is (for reloading saved markers).
        """
        if r is None:
            r = self._compute_radius_for_current_zoom()

        ellipse = self.scene().addEllipse(
            x - r, y - r, 2 * r, 2 * r,
            QPen(Qt.GlobalColor.red)
        )
        self.markers.append({"item": ellipse, "x": x, "y": y, "r": r})

    def remove_last_marker(self):
        """Remove the last marker, if any."""
        if self.markers:
            last = self.markers.pop()
            self.scene().removeItem(last["item"])

    def find_nearest_marker(self, x, y, max_distance: float):
        """
        Find the nearest marker to (x, y) in scene coordinates,
        within max_distance (also in scene coordinates).
        Returns (index, marker_dict) or (-1, None) if none is close enough.
        """
        if not self.markers:
            return -1, None

        best_idx = -1
        best_dist2 = max_distance * max_distance

        for i, m in enumerate(self.markers):
            dx = m["x"] - x
            dy = m["y"] - y
            d2 = dx * dx + dy * dy
            if d2 <= best_dist2:
                best_dist2 = d2
                best_idx = i

        if best_idx == -1:
            return -1, None
        return best_idx, self.markers[best_idx]

    # ---------- Zoom handling (via slider) ----------

    def set_zoom_factor(self, factor_relative: float):
        """
        Set the zoom to a relative factor with respect to base_zoom.
        factor_relative = 1.0 means "fit to view" (base zoom).
        """
        # clamp
        factor = max(self.min_factor, min(self.max_factor, factor_relative))

        new_zoom = self.base_zoom * factor

        # compute ratio to current zoom and scale relatively, preserving pan
        ratio = new_zoom / self.current_zoom if self.current_zoom != 0 else 1.0
        self.scale(ratio, ratio)

        self.current_zoom = new_zoom
        self.current_factor = factor

    # ---------- Mouse interaction ----------

    def mousePressEvent(self, event):
        # Left button: begin possible drag / click
        if event.button() == Qt.MouseButton.LeftButton and self.image_item is not None:
            self.left_mouse_pressed = True
            self.moved_since_press = False
            self.press_pos = event.position().toPoint()

            # enable hand dragging
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(event)

        # Right button: show context menu to delete nearest marker
        elif event.button() == Qt.MouseButton.RightButton and self.image_item is not None:
            pos_scene = self.mapToScene(event.position().toPoint())
            x, y = pos_scene.x(), pos_scene.y()

            idx, marker = self.find_nearest_marker(x, y, self.pick_radius)
            if marker is not None:
                menu = QMenu(self)
                delete_action = menu.addAction("Delete marker")
                chosen = menu.exec(event.globalPosition().toPoint())
                if chosen == delete_action:
                    removed = self.markers.pop(idx)
                    self.scene().removeItem(removed["item"])
                    if self.delete_callback is not None:
                        self.delete_callback(removed["x"], removed["y"], removed["r"])
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.left_mouse_pressed and self.press_pos is not None:
            if (event.position().toPoint() - self.press_pos).manhattanLength() > self.click_threshold:
                self.moved_since_press = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.image_item is not None:
            super().mouseReleaseEvent(event)

            # stop hand dragging
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

            # treat as click if mouse didn't move much
            if not self.moved_since_press:
                pos_scene = self.mapToScene(event.position().toPoint())
                x, y = pos_scene.x(), pos_scene.y()
                r = self._compute_radius_for_current_zoom()
                self.add_marker(x, y, r)
                if self.click_callback is not None:
                    self.click_callback(x, y, r)

            # reset state
            self.left_mouse_pressed = False
            self.press_pos = None
            self.moved_since_press = False
        else:
            super().mouseReleaseEvent(event)




class LabelingApp(QMainWindow):
    def __init__(self, folder_path: Path):
        super().__init__()

        self.folder = folder_path
        self.image_paths = self.load_image_list(self.folder)
        if not self.image_paths:
            raise ValueError("No .tif images found in folder")

        self.current_index = 0
        # image_name -> list[(x, y)]
        self.labels = {p.name: [] for p in self.image_paths}


        # try to load existing labels if present
        self.load_existing_labels()

        self.init_ui()
        self.load_image()

    # ---------- Setup ----------

    def init_ui(self):
        self.setWindowTitle("Cell Labeling Tool (PyQt6)")

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Image view
        self.image_view = ImageView()
        self.image_view.click_callback = self.on_image_click
        self.image_view.delete_callback = self.on_marker_deleted

        main_layout.addWidget(self.image_view)

        # Info + controls
        bottom_layout = QHBoxLayout()

        self.info_label = QLabel("Image: - | Cells: 0")
        bottom_layout.addWidget(self.info_label)

        bottom_layout.addStretch(1)

        # Zoom slider: 10% .. 400% of base zoom
        zoom_label = QLabel("Zoom")
        bottom_layout.addWidget(zoom_label)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 400)   # 10% to 400%
        self.zoom_slider.setValue(100)       # start at 100% (base zoom)
        self.zoom_slider.setFixedWidth(200)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        bottom_layout.addWidget(self.zoom_slider)

        # Undo
        self.undo_button = QPushButton("Undo last click")
        self.undo_button.clicked.connect(self.undo_last_click)
        bottom_layout.addWidget(self.undo_button)

        # Navigation
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_image)
        bottom_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_image)
        bottom_layout.addWidget(self.next_button)

        # Save
        self.save_button = QPushButton("Save labels")
        self.save_button.clicked.connect(self.save_labels)
        bottom_layout.addWidget(self.save_button)


        main_layout.addLayout(bottom_layout)

        # Toolbar / menu actions
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        open_action = QAction("Open folder", self)
        open_action.triggered.connect(self.choose_folder)
        toolbar.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_labels)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        toolbar.addAction(save_action)

        undo_action = QAction("Undo", self)
        undo_action.triggered.connect(self.undo_last_click)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        toolbar.addAction(undo_action)

        # Status bar
        self.statusBar().showMessage("Ready")

    @staticmethod
    def load_image_list(folder: Path):
        """
        Return a sorted list of valid .tif images in the folder.

        - Skips macOS resource fork files (names starting with '._')
        - Sorts by the time code pattern 00d00h00m in the filename.
        """

        # Get all .tif files but skip '._' junk files
        image_paths = [
            p for p in folder.glob("*.tif")
            if not p.name.startswith("._")
        ]

        # Sort using the time key extracted from filename
        image_paths = sorted(image_paths, key=time_key_from_path)

        return image_paths


    # ---------- Folder handling ----------

    def choose_folder(self):
        folder_str = QFileDialog.getExistingDirectory(self, "Choose image folder", str(self.folder))
        if not folder_str:
            return
        folder = Path(folder_str)
        image_paths = self.load_image_list(folder)
        if not image_paths:
            QMessageBox.warning(self, "No images", "No .tif images found in the selected folder.")
            return

        self.folder = folder
        self.image_paths = image_paths
        self.current_index = 0
        self.labels = {p.name: [] for p in self.image_paths}
        self.load_existing_labels()
        self.load_image()
        self.statusBar().showMessage(f"Loaded folder: {self.folder}", 5000)

    # ---------- Labels I/O ----------

    def labels_csv_path(self) -> Path:
        return self.folder / "cell_labels.csv"

    def load_existing_labels(self):
        """
        If cell_labels.csv exists in the folder, load it and populate self.labels.
        Supports both old format (no 'r' column) and new format with 'r'.
        """
        csv_path = self.labels_csv_path()
        if not csv_path.exists():
            return

        try:
            with csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                has_r = ("r" in reader.fieldnames) if reader.fieldnames else False

                for row in reader:
                    img_name = row["image_name"]
                    if img_name not in self.labels:
                        continue
                    x = float(row["x"])
                    y = float(row["y"])
                    if has_r:
                        r = float(row["r"])
                    else:
                        r = BASE_MARKER_RADIUS

                    self.labels[img_name].append((x, y, r))

            self.statusBar().showMessage(f"Loaded existing labels from {csv_path}", 5000)
        except Exception as e:
            QMessageBox.warning(self, "Load error", f"Could not load existing labels:\n{e}")


    def save_labels(self):
        """
        Save all labels to cell_labels.csv in the current folder.
        Format: image_name,cell_id,x,y,r
        """
        csv_path = self.labels_csv_path()
        try:
            with csv_path.open("w", newline="") as f:
                fieldnames = ["image_name", "cell_id", "x", "y", "r"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for img_name, points in self.labels.items():
                    for i, (x, y, r) in enumerate(points):
                        writer.writerow({
                            "image_name": img_name,
                            "cell_id": i,
                            "x": x,
                            "y": y,
                            "r": r,
                        })
            self.statusBar().showMessage(f"Saved labels to {csv_path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Could not save labels:\n{e}")


    # ---------- Image navigation / drawing ----------
    
    def on_marker_deleted(self, x, y, r):
        """
        Called by ImageView when a marker is deleted via right-click context menu.
        Remove the nearest stored label for the current image.
        """
        img_name = self.image_paths[self.current_index].name
        points = self.labels.get(img_name, [])
        if not points:
            return

        # find nearest stored point to (x, y)
        best_idx = None
        best_dist2 = None
        for i, (px, py, pr) in enumerate(points):
            dx = px - x
            dy = py - y
            d2 = dx * dx + dy * dy
            if best_dist2 is None or d2 < best_dist2:
                best_dist2 = d2
                best_idx = i

        if best_idx is not None:
            points.pop(best_idx)
            self.update_info_label()



    def load_image(self):
        """Load the current image and draw its markers."""
        if not self.image_paths:
            return

        img_path = self.image_paths[self.current_index]
        pil_img = Image.open(img_path)
        pixmap = pil_to_qpixmap(pil_img)

        self.image_view.set_image(pixmap)

        # re-draw markers for this image
        self.image_view.clear_markers()
        img_name = img_path.name
        for (x, y, r) in self.labels[img_name]:
            self.image_view.add_marker(x, y, r)

        # apply current zoom factor from slider
        if hasattr(self, "zoom_slider"):
            factor = self.zoom_slider.value() / 100.0
            self.image_view.set_zoom_factor(factor)

        self.update_info_label()



    def update_info_label(self):
        img_name = self.image_paths[self.current_index].name
        n_cells = len(self.labels[img_name])
        self.info_label.setText(
            f"Image: {img_name} ({self.current_index + 1}/{len(self.image_paths)}) | Cells: {n_cells}"
        )
        
    def on_zoom_slider_changed(self, value: int):
        """
        Called when the zoom slider moves.
        Slider value represents percentage of base zoom.
        """
        factor = value / 100.0  # 100 -> 1.0, 200 -> 2.0, etc.
        self.image_view.set_zoom_factor(factor)


    def on_image_click(self, x, y, r):
        img_name = self.image_paths[self.current_index].name
        self.labels[img_name].append((x, y, r))
        self.update_info_label()

    def undo_last_click(self):
        """Undo last click for the current image."""
        img_name = self.image_paths[self.current_index].name
        if self.labels[img_name]:
            self.labels[img_name].pop()       # remove from data
            self.image_view.remove_last_marker()  # remove from view
            self.update_info_label()
        else:
            self.statusBar().showMessage("No markers to undo for this image.", 3000)

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_image()


def main():
    app = QApplication(sys.argv)

    # Choose initial folder
    folder_str = QFileDialog.getExistingDirectory(None, "Choose initial image folder")
    if not folder_str:
        sys.exit(0)

    folder = Path(folder_str)
    try:
        win = LabelingApp(folder)
    except ValueError as e:
        QMessageBox.critical(None, "Error", str(e))
        sys.exit(1)

    win.resize(1000, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
