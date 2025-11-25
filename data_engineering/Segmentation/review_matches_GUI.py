
# """
# GUI to manually review and correct matches between ground-truth clicks
# and segmentation labels.

# This version uses PIL instead of tifffile to load TIFF images, so it
# does not require the 'imagecodecs' package for LZW-compressed TIFFs.

# Inputs:
#     - images_dir: folder with raw images (same image_name as in matches.csv)
#     - labels_dir: folder with label masks (0 background, 1..N segments)
#     - matches_csv: CSV from evaluate_segmentation_points.py with columns:
#         image_name, cell_id, x, y, matched_label, distance_to_segment

# Usage:
#     python review_matches_gui.py \
#         --images-dir /path/to/raw_images \
#         --labels-dir /path/to/labels \
#         --matches-csv eval_results/matches.csv \
#         --label-suffix "_labels.tif"

# Actions:
#     - Left panel: image + label overlay + click markers (green=matched, red=unmatched).
#     - Right panel:
#         * list of GT clicks (rows in matches.csv) for the current image
#         * "Previous" / "Next" image buttons
#         * "Set no match" button to mark the selected click as unmatched
#         * "Save CSV" button to write corrected matches back to matches.csv

#     - Right-click on the image:
#         * If a click is selected in the list, assign that click to the segment
#           under the cursor (label id from label mask).
#         * The GUI enforces that each segment label is assigned to at most one
#           GT click per image: if another click already uses that label, it is
#           automatically set to -1 (unmatched).
# """

# import sys
# import argparse
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from PIL import Image

# from PyQt6.QtCore import Qt, QPointF, QRectF
# from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor
# from PyQt6.QtWidgets import (
#     QApplication,
#     QMainWindow,
#     QWidget,
#     QVBoxLayout,
#     QHBoxLayout,
#     QLabel,
#     QPushButton,
#     QListWidget,
#     QListWidgetItem,
#     QGraphicsView,
#     QGraphicsScene,
#     QGraphicsEllipseItem,
# )


# # ---------------------------------------------------------------------
# # ImageView: displays overlay and forwards right-clicks
# # ---------------------------------------------------------------------

# class ImageView(QGraphicsView):
#     """
#     QGraphicsView to display an image and markers.
#     For right-clicks, it forwards image coordinates to a callback.
#     """

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setScene(QGraphicsScene(self))
#         self.image_item = None
#         self.image_width = 0
#         self.image_height = 0

#         # global_idx -> marker item
#         self.marker_items = {}

#         # callback: function(x, y) in image coordinates
#         self.right_click_callback = None

#     def set_pixmap(self, pixmap: QPixmap):
#         """Set the base image pixmap."""
#         self.scene().clear()
#         self.marker_items = {}
#         self.image_item = self.scene().addPixmap(pixmap)
#         self.setSceneRect(self.image_item.boundingRect())
#         self.image_width = pixmap.width()
#         self.image_height = pixmap.height()

#     def clear_markers(self):
#         for item in self.marker_items.values():
#             self.scene().removeItem(item)
#         self.marker_items = {}

#     def add_marker(self, x, y, global_idx, matched: bool, selected: bool = False):
#         """
#         Add a circular marker at (x, y) image coords.

#         matched: True => green, False => red
#         selected: if True, slightly larger and thicker border
#         """
#         radius = 5 if selected else 3
#         pen_width = 2 if selected else 1

#         color = QColor(0, 200, 0) if matched else QColor(200, 0, 0)

#         pen = QPen(color)
#         pen.setWidth(pen_width)
#         brush = QBrush(Qt.BrushStyle.NoBrush)

#         item = self.scene().addEllipse(
#             x - radius,
#             y - radius,
#             2 * radius,
#             2 * radius,
#             pen,
#             brush,
#         )
#         item.setZValue(20)  # above image

#         self.marker_items[global_idx] = item

#     def update_marker_style(self, global_idx, matched: bool, selected: bool):
#         """Update color/size of an existing marker."""
#         item = self.marker_items.get(global_idx)
#         if item is None:
#             return

#         radius = 5 if selected else 3
#         pen_width = 2 if selected else 1
#         color = QColor(0, 200, 0) if matched else QColor(200, 0, 0)
#         pen = QPen(color)
#         pen.setWidth(pen_width)

#         rect = item.rect()
#         cx = rect.center().x()
#         cy = rect.center().y()

#         new_rect = QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius)
#         item.setRect(new_rect)
#         item.setPen(pen)

#     def mousePressEvent(self, event):
#         if event.button() == Qt.MouseButton.RightButton and self.image_item is not None:
#             pos_view = event.position().toPoint()
#             pos_scene = self.mapToScene(pos_view)
#             x = pos_scene.x()
#             y = pos_scene.y()
#             if 0 <= x < self.image_width and 0 <= y < self.image_height:
#                 if self.right_click_callback is not None:
#                     self.right_click_callback(float(x), float(y))
#         else:
#             super().mousePressEvent(event)


# # ---------------------------------------------------------------------
# # Utility to build overlay image
# # ---------------------------------------------------------------------

# def build_overlay_image(raw: np.ndarray, labels: np.ndarray) -> np.ndarray:
#     """
#     Build an RGB overlay image from raw and label map.

#     raw    : 2D array (H,W) or 3D (H,W,C); will be converted to grayscale.
#     labels : 2D array (H,W) of ints (0 background, 1..N segments)

#     Returns:
#         uint8 array (H,W,3)
#     """
#     # raw -> grayscale float in [0,1]
#     if raw.ndim == 3:
#         raw_gray = raw.mean(axis=-1)
#     else:
#         raw_gray = raw

#     raw_gray = raw_gray.astype(np.float32)
#     if raw_gray.max() > 0:
#         raw_gray /= raw_gray.max()
#     raw_rgb = np.stack([raw_gray, raw_gray, raw_gray], axis=-1)  # H,W,3

#     h, w = raw_gray.shape
#     overlay = raw_rgb.copy()

#     max_label = int(labels.max())
#     if max_label == 0:
#         return (overlay * 255).astype(np.uint8)

#     rng = np.random.default_rng(42)  # deterministic colors
#     colors = rng.random((max_label + 1, 3), dtype=np.float32)
#     colors[0] = 0.0  # background

#     for lbl in range(1, max_label + 1):
#         mask = labels == lbl
#         if not np.any(mask):
#             continue
#         color = colors[lbl]
#         overlay[mask] = 0.6 * color + 0.4 * overlay[mask]

#     overlay = np.clip(overlay, 0.0, 1.0)
#     return (overlay * 255).astype(np.uint8)


# def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
#     """
#     Convert HxWx3 uint8 numpy array to QPixmap.
#     """
#     h, w, _ = img.shape
#     qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
#     return QPixmap.fromImage(qimg)


# # ---------------------------------------------------------------------
# # Main window
# # ---------------------------------------------------------------------

# class ReviewWindow(QMainWindow):
#     def __init__(self, images_dir: Path, labels_dir: Path, matches_csv: Path, label_suffix: str = ""):
#         super().__init__()

#         self.images_dir = images_dir
#         self.labels_dir = labels_dir
#         self.matches_csv = matches_csv
#         self.label_suffix = label_suffix

#         # Load matches
#         self.matches_df = pd.read_csv(matches_csv)
#         required_cols = {"image_name", "cell_id", "x", "y", "matched_label"}
#         if not required_cols.issubset(self.matches_df.columns):
#             raise ValueError(f"matches.csv must contain columns {required_cols}, got {self.matches_df.columns}")

#         self.image_names = sorted(self.matches_df["image_name"].unique())
#         if not self.image_names:
#             raise ValueError("No images found in matches.csv")

#         # Mapping: image_name -> list of global row indices in matches_df
#         self.image_to_indices = {
#             name: self.matches_df.index[self.matches_df["image_name"] == name].tolist()
#             for name in self.image_names
#         }

#         self.current_image_idx = 0
#         self.current_image_name = self.image_names[0]
#         self.current_label_img = None  # numpy 2D
#         self.current_raw = None        # numpy 2D or 3D
#         self.current_indices = []      # list of row indices for current image
#         self.current_selected_global_idx = None

#         self.init_ui()
#         self.load_current_image()

#     def init_ui(self):
#         self.setWindowTitle("Segmentation Matches Review")

#         central = QWidget()
#         self.setCentralWidget(central)

#         main_layout = QHBoxLayout(central)

#         # Left: image view
#         self.image_view = ImageView()
#         self.image_view.right_click_callback = self.on_image_right_click
#         main_layout.addWidget(self.image_view, stretch=3)

#         # Right: controls
#         right_layout = QVBoxLayout()

#         self.image_label = QLabel("Image: -")
#         right_layout.addWidget(self.image_label)

#         self.matches_list = QListWidget()
#         self.matches_list.currentItemChanged.connect(self.on_list_selection_changed)
#         right_layout.addWidget(self.matches_list, stretch=1)

#         # Buttons
#         button_row = QHBoxLayout()

#         self.prev_button = QPushButton("Previous image")
#         self.prev_button.clicked.connect(self.prev_image)
#         button_row.addWidget(self.prev_button)

#         self.next_button = QPushButton("Next image")
#         self.next_button.clicked.connect(self.next_image)
#         button_row.addWidget(self.next_button)

#         right_layout.addLayout(button_row)

#         button_row2 = QHBoxLayout()

#         self.no_match_button = QPushButton("Set no match")
#         self.no_match_button.clicked.connect(self.set_no_match_for_selected)
#         button_row2.addWidget(self.no_match_button)

#         self.save_button = QPushButton("Save CSV")
#         self.save_button.clicked.connect(self.save_matches_csv)
#         button_row2.addWidget(self.save_button)

#         right_layout.addLayout(button_row2)

#         # info label
#         self.info_label = QLabel(
#             "Tips:\n"
#             "1) Select a click in the list.\n"
#             "2) Right-click on a segment in the image\n"
#             "   to assign that label to the selected click.\n"
#             "3) 'Set no match' marks the selected click as unmatched (-1).\n"
#             "4) 'Save CSV' overwrites matches.csv."
#         )
#         self.info_label.setWordWrap(True)
#         right_layout.addWidget(self.info_label)

#         main_layout.addLayout(right_layout, stretch=1)

#     # -------------------------
#     # Loading images & labels
#     # -------------------------

#     def load_raw_image(self, image_name: str):
#         path = self.images_dir / image_name
#         if not path.exists():
#             raise FileNotFoundError(f"Raw image file not found: {path}")
#         img = Image.open(str(path))
#         arr = np.array(img)
#         return arr

#     def load_label_image(self, image_name: str):
#         p1 = self.labels_dir / image_name
#         stem = Path(image_name).stem
#         p2 = self.labels_dir / f"{stem}{self.label_suffix}"

#         for p in (p1, p2):
#             if p.exists():
#                 img = Image.open(str(p))
#                 arr = np.array(img)
#                 # if bits are stored in multiple channels, take first
#                 if arr.ndim == 3:
#                     arr = arr[..., 0]
#                 if arr.ndim != 2:
#                     raise ValueError(f"Label image must be 2D, got {arr.shape} for {p}")
#                 return arr.astype(np.int32)

#         raise FileNotFoundError(f"No label image found for {image_name} in {self.labels_dir}")

#     def load_current_image(self):
#         self.current_image_name = self.image_names[self.current_image_idx]
#         self.current_indices = self.image_to_indices[self.current_image_name]

#         self.image_label.setText(
#             f"Image: {self.current_image_name}  "
#             f"({self.current_image_idx+1}/{len(self.image_names)})"
#         )

#         # load raw + labels
#         self.current_raw = self.load_raw_image(self.current_image_name)
#         self.current_label_img = self.load_label_image(self.current_image_name)

#         # make overlay
#         overlay = build_overlay_image(self.current_raw, self.current_label_img)
#         pixmap = numpy_to_qpixmap(overlay)
#         self.image_view.set_pixmap(pixmap)

#         # populate list + markers
#         self.populate_list_and_markers()

#     # -------------------------
#     # UI helpers
#     # -------------------------

#     def populate_list_and_markers(self):
#         self.matches_list.clear()
#         self.image_view.clear_markers()
#         self.current_selected_global_idx = None

#         rows = self.matches_df.loc[self.current_indices]

#         for global_idx, row in rows.iterrows():
#             cell_id = row["cell_id"]
#             x = row["x"]
#             y = row["y"]
#             matched_label = row["matched_label"]

#             matched = (matched_label != -1)
#             dist = row.get("distance_to_segment", np.nan)

#             text = f"cell_id={cell_id} | match={matched_label} | dist={dist:.2f}"
#             item = QListWidgetItem(text)
#             color = QColor(0, 150, 0) if matched else QColor(200, 0, 0)
#             item.setForeground(color)
#             item.setData(Qt.ItemDataRole.UserRole, global_idx)  # store global index
#             self.matches_list.addItem(item)

#             # add marker
#             self.image_view.add_marker(x, y, global_idx, matched=matched, selected=False)

#     def update_list_item_for_row(self, global_idx):
#         # find item with given global_idx
#         for i in range(self.matches_list.count()):
#             item = self.matches_list.item(i)
#             idx = item.data(Qt.ItemDataRole.UserRole)
#             if idx == global_idx:
#                 row = self.matches_df.loc[global_idx]
#                 matched_label = row["matched_label"]
#                 matched = (matched_label != -1)
#                 dist = row.get("distance_to_segment", np.nan)
#                 text = f"cell_id={row['cell_id']} | match={matched_label} | dist={dist:.2f}"
#                 item.setText(text)
#                 color = QColor(0, 150, 0) if matched else QColor(200, 0, 0)
#                 item.setForeground(color)

#                 # update marker style
#                 selected = (self.current_selected_global_idx == global_idx)
#                 self.image_view.update_marker_style(global_idx, matched=matched, selected=selected)
#                 break

#     # -------------------------
#     # Navigation
#     # -------------------------

#     def prev_image(self):
#         if self.current_image_idx > 0:
#             self.current_image_idx -= 1
#             self.load_current_image()

#     def next_image(self):
#         if self.current_image_idx < len(self.image_names) - 1:
#             self.current_image_idx += 1
#             self.load_current_image()

#     # -------------------------
#     # List selection
#     # -------------------------

#     def on_list_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
#         # un-highlight the previously selected marker
#         if previous is not None:
#             prev_idx = previous.data(Qt.ItemDataRole.UserRole)
#             if prev_idx is not None:
#                 row = self.matches_df.loc[prev_idx]
#                 matched = (row["matched_label"] != -1)
#                 self.image_view.update_marker_style(prev_idx, matched=matched, selected=False)

#         # highlight the newly selected marker
#         if current is not None:
#             cur_idx = current.data(Qt.ItemDataRole.UserRole)
#             if cur_idx is not None:
#                 self.current_selected_global_idx = cur_idx
#                 row = self.matches_df.loc[cur_idx]
#                 matched = (row["matched_label"] != -1)
#                 self.image_view.update_marker_style(cur_idx, matched=matched, selected=True)
#         else:
#             self.current_selected_global_idx = None

#     # -------------------------
#     # Manual edits
#     # -------------------------

#     def set_no_match_for_selected(self):
#         if self.current_selected_global_idx is None:
#             return
#         idx = self.current_selected_global_idx
#         self.matches_df.loc[idx, "matched_label"] = -1
#         # distance might not matter now
#         self.matches_df.loc[idx, "distance_to_segment"] = np.inf
#         self.update_list_item_for_row(idx)

#     def on_image_right_click(self, x: float, y: float):
#         """
#         Called when user right-clicks on the image.
#         If a click is selected, we assign it to the segment under (x,y).
#         """
#         if self.current_selected_global_idx is None:
#             return

#         if self.current_label_img is None:
#             return

#         # map to integer pixel coordinates
#         ix = int(round(x))
#         iy = int(round(y))

#         h, w = self.current_label_img.shape
#         if not (0 <= ix < w and 0 <= iy < h):
#             return

#         label_id = int(self.current_label_img[iy, ix])
#         if label_id == 0:
#             # background, ignore
#             return

#         # enforce 1:1: if another click in this image already has this label,
#         # unset it (set to -1)
#         for global_idx in self.current_indices:
#             if global_idx == self.current_selected_global_idx:
#                 continue
#             row = self.matches_df.loc[global_idx]
#             if row["matched_label"] == label_id:
#                 self.matches_df.loc[global_idx, "matched_label"] = -1
#                 self.matches_df.loc[global_idx, "distance_to_segment"] = np.inf
#                 self.update_list_item_for_row(global_idx)

#         # assign label to selected click
#         idx = self.current_selected_global_idx
#         row = self.matches_df.loc[idx]
#         x_pt = row["x"]
#         y_pt = row["y"]

#         # recompute distance from point to nearest pixel in this label
#         ys, xs = np.where(self.current_label_img == label_id)
#         if xs.size > 0:
#             coords = np.stack([xs, ys], axis=1).astype(np.float32)
#             pt = np.array([x_pt, y_pt], dtype=np.float32)
#             d2 = np.sum((coords - pt) ** 2, axis=1)
#             dist = float(np.sqrt(d2.min()))
#         else:
#             dist = np.inf

#         self.matches_df.loc[idx, "matched_label"] = label_id
#         self.matches_df.loc[idx, "distance_to_segment"] = dist
#         self.update_list_item_for_row(idx)

#     def save_matches_csv(self):
#         self.matches_df.to_csv(self.matches_csv, index=False)
#         self.statusBar().showMessage(f"Saved updated matches to {self.matches_csv}", 5000)


# # ---------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------

# def parse_args():
#     ap = argparse.ArgumentParser(description="GUI to manually review and correct segmentation matches.")
#     ap.add_argument("--images-dir", type=str, required=True,
#                     help="Directory with raw images.")
#     ap.add_argument("--labels-dir", type=str, required=True,
#                     help="Directory with label masks.")
#     ap.add_argument("--matches-csv", type=str, required=True,
#                     help="Path to matches.csv from evaluation script.")
#     ap.add_argument("--label-suffix", type=str, default="",
#                     help="Optional suffix for label files, e.g. '_labels.tif'.")
#     return ap.parse_args()


# def main():
#     args = parse_args()
#     app = QApplication(sys.argv)
#     win = ReviewWindow(
#         images_dir=Path(args.images_dir),
#         labels_dir=Path(args.labels_dir),
#         matches_csv=Path(args.matches_csv),
#         label_suffix=args.label_suffix,
#     )
#     win.resize(1400, 800)
#     win.show()
#     sys.exit(app.exec())


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
GUI to manually review and correct matches between ground-truth clicks
and segmentation labels.

Features:
  - Uses PIL (not tifffile) to load TIFFs (works with LZW compression).
  - Zoom in/out via buttons (no wheel zoom).
  - Left-drag to pan.
  - Sidebar list of clicks; selecting one highlights its marker (yellow).
  - Clicking a marker in the image selects the corresponding list entry.
    Clicking the same marker again deselects.
  - Right-click on a segment assigns that label to the selected click.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QPen,
    QBrush,
    QColor,
    QPainter,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QGraphicsView,
    QGraphicsScene,
)

# ---------------------------------------------------------------------
# Custom list widget to allow "click again to deselect"
# ---------------------------------------------------------------------

class MatchListWidget(QListWidget):
    """
    QListWidget where clicking the already-selected item toggles it off
    (clears selection).
    """

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if item is not None and item == self.currentItem():
            # clicking the same item -> clear selection
            self.clearSelection()
            event.accept()
            return
        super().mousePressEvent(event)


# ---------------------------------------------------------------------
# ImageView: overlay, markers, zoom buttons, pan, right/left click callbacks
# ---------------------------------------------------------------------

class ImageView(QGraphicsView):
    """
    QGraphicsView to display an image and markers.

    - Zoom via zoom_in / zoom_out methods (buttons).
    - Left-drag pans (ScrollHandDrag).
    - Right-click forwards image coordinates to right_click_callback.
    - Left-click on a marker forwards the associated global index via
      marker_click_callback.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.image_item = None
        self.image_width = 0
        self.image_height = 0

        # global_idx -> marker item
        self.marker_items = {}

        # callbacks
        self.right_click_callback = None
        self.marker_click_callback = None

        # zoom handling
        self._zoom = 0
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def reset_zoom(self):
        self._zoom = 0
        self.resetTransform()

    def set_pixmap(self, pixmap: QPixmap):
        """Set the base image pixmap."""
        self.scene().clear()
        self.marker_items = {}
        self.image_item = self.scene().addPixmap(pixmap)
        self.setSceneRect(self.image_item.boundingRect())
        self.image_width = pixmap.width()
        self.image_height = pixmap.height()
        self.reset_zoom()

    def clear_markers(self):
        for item in self.marker_items.values():
            self.scene().removeItem(item)
        self.marker_items = {}

    def add_marker(self, x, y, global_idx, matched: bool, selected: bool = False):
        """
        Add a circular marker at (x, y) image coords.

        matched: True => green, False => red
        selected: if True, larger and yellow outline.
        """
        radius = 5 if selected else 3
        pen_width = 3 if selected else 1

        if selected:
            color = QColor(255, 255, 0)  # bright yellow
        else:
            color = QColor(0, 200, 0) if matched else QColor(200, 0, 0)

        pen = QPen(color)
        pen.setWidth(pen_width)
        brush = QBrush(Qt.BrushStyle.NoBrush)

        item = self.scene().addEllipse(
            x - radius,
            y - radius,
            2 * radius,
            2 * radius,
            pen,
            brush,
        )
        item.setZValue(20)  # above image

        self.marker_items[global_idx] = item

    def update_marker_style(self, global_idx, matched: bool, selected: bool):
        """Update color/size of an existing marker."""
        item = self.marker_items.get(global_idx)
        if item is None:
            return

        radius = 5 if selected else 3
        pen_width = 3 if selected else 1

        if selected:
            color = QColor(255, 255, 0)  # yellow
        else:
            color = QColor(0, 200, 0) if matched else QColor(200, 0, 0)

        pen = QPen(color)
        pen.setWidth(pen_width)

        rect = item.rect()
        cx = rect.center().x()
        cy = rect.center().y()

        new_rect = QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius)
        item.setRect(new_rect)
        item.setPen(pen)

    # zoom controlled by buttons
    def zoom_in(self):
        if self.image_item is None:
            return
        if self._zoom >= 30:
            return
        self.scale(1.2, 1.2)
        self._zoom += 1

    def zoom_out(self):
        if self.image_item is None:
            return
        if self._zoom <= -10:
            return
        self.scale(1 / 1.2, 1 / 1.2)
        self._zoom -= 1

    def mousePressEvent(self, event):
        pos_view = event.position().toPoint()

        if event.button() == Qt.MouseButton.RightButton and self.image_item is not None:
            # right-click: assign segment
            pos_scene = self.mapToScene(pos_view)
            x = pos_scene.x()
            y = pos_scene.y()
            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                if self.right_click_callback is not None:
                    self.right_click_callback(float(x), float(y))
            # let base class handle drag, etc.
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton and self.image_item is not None:
            # left-click: check if we clicked a marker
            item = self.itemAt(pos_view)
            if item is not None and item in self.marker_items.values():
                # find which marker
                for gid, m_item in self.marker_items.items():
                    if m_item is item:
                        if self.marker_click_callback is not None:
                            self.marker_click_callback(gid)
                        break

        super().mousePressEvent(event)

    def wheelEvent(self, event):
        """
        Disable wheel-based zoom; let QGraphicsView handle scrollbars normally.
        """
        QGraphicsView.wheelEvent(self, event)


# ---------------------------------------------------------------------
# Utility to build overlay image
# ---------------------------------------------------------------------

def build_overlay_image(raw: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Build an RGB overlay image from raw and label map.

    raw    : 2D array (H,W) or 3D (H,W,C); will be converted to grayscale.
    labels : 2D array (H,W) of ints (0 background, 1..N segments)

    Returns:
        uint8 array (H,W,3)
    """
    # raw -> grayscale float in [0,1]
    if raw.ndim == 3:
        raw_gray = raw.mean(axis=-1)
    else:
        raw_gray = raw

    raw_gray = raw_gray.astype(np.float32)
    if raw_gray.max() > 0:
        raw_gray /= raw_gray.max()
    raw_rgb = np.stack([raw_gray, raw_gray, raw_gray], axis=-1)  # H,W,3

    overlay = raw_rgb.copy()

    max_label = int(labels.max())
    if max_label == 0:
        return (overlay * 255).astype(np.uint8)

    rng = np.random.default_rng(42)  # deterministic colors
    colors = rng.random((max_label + 1, 3), dtype=np.float32)
    colors[0] = 0.0  # background

    for lbl in range(1, max_label + 1):
        mask = labels == lbl
        if not np.any(mask):
            continue
        color = colors[lbl]
        overlay[mask] = 0.6 * color + 0.4 * overlay[mask]

    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    """
    Convert HxWx3 uint8 numpy array to QPixmap.
    """
    h, w, _ = img.shape
    qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------

class ReviewWindow(QMainWindow):
    def __init__(self, images_dir: Path, labels_dir: Path, matches_csv: Path, label_suffix: str = ""):
        super().__init__()

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.matches_csv = matches_csv
        self.label_suffix = label_suffix

        # Load matches
        self.matches_df = pd.read_csv(matches_csv)
        required_cols = {"image_name", "cell_id", "x", "y", "matched_label"}
        if not required_cols.issubset(self.matches_df.columns):
            raise ValueError(f"matches.csv must contain columns {required_cols}, got {self.matches_df.columns}")

        self.image_names = sorted(self.matches_df["image_name"].unique())
        if not self.image_names:
            raise ValueError("No images found in matches.csv")

        # Mapping: image_name -> list of global row indices in matches_df
        self.image_to_indices = {
            name: self.matches_df.index[self.matches_df["image_name"] == name].tolist()
            for name in self.image_names
        }

        self.current_image_idx = 0
        self.current_image_name = self.image_names[0]
        self.current_label_img = None  # numpy 2D
        self.current_raw = None        # numpy 2D or 3D
        self.current_indices = []      # list of row indices for current image
        self.current_selected_global_idx = None

        # map from global_idx -> row index in the list widget for current image
        self.global_to_row_index = {}

        self.init_ui()
        self.load_current_image()

    def init_ui(self):
        self.setWindowTitle("Segmentation Matches Review")

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # Left: image view
        self.image_view = ImageView()
        self.image_view.right_click_callback = self.on_image_right_click
        self.image_view.marker_click_callback = self.on_marker_clicked
        main_layout.addWidget(self.image_view, stretch=3)

        # Right: controls
        right_layout = QVBoxLayout()

        self.image_label = QLabel("Image: -")
        right_layout.addWidget(self.image_label)

        self.matches_list = MatchListWidget()
        self.matches_list.currentItemChanged.connect(self.on_list_selection_changed)
        right_layout.addWidget(self.matches_list, stretch=1)

        # Navigation buttons
        nav_row = QHBoxLayout()

        self.prev_button = QPushButton("Previous image")
        self.prev_button.clicked.connect(self.prev_image)
        nav_row.addWidget(self.prev_button)

        self.next_button = QPushButton("Next image")
        self.next_button.clicked.connect(self.next_image)
        nav_row.addWidget(self.next_button)

        right_layout.addLayout(nav_row)

        # Zoom buttons
        zoom_row = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom in")
        self.zoom_in_button.clicked.connect(self.image_view.zoom_in)
        zoom_row.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom out")
        self.zoom_out_button.clicked.connect(self.image_view.zoom_out)
        zoom_row.addWidget(self.zoom_out_button)

        right_layout.addLayout(zoom_row)

        # Edit / save buttons
        edit_row = QHBoxLayout()

        self.no_match_button = QPushButton("Set no match")
        self.no_match_button.clicked.connect(self.set_no_match_for_selected)
        edit_row.addWidget(self.no_match_button)

        self.save_button = QPushButton("Save CSV")
        self.save_button.clicked.connect(self.save_matches_csv)
        edit_row.addWidget(self.save_button)

        right_layout.addLayout(edit_row)

        # info label
        self.info_label = QLabel(
            "Tips:\n"
            "1) Select a click in the list OR click its marker in the image.\n"
            "2) Right-click on a segment in the image\n"
            "   to assign that label to the selected click.\n"
            "3) Click the same list item or marker again to deselect.\n"
            "4) Zoom in/out with buttons. Left-drag to pan.\n"
            "5) 'Set no match' marks the selected click as unmatched (-1).\n"
            "6) 'Save CSV' overwrites matches.csv."
        )
        self.info_label.setWordWrap(True)
        right_layout.addWidget(self.info_label)

        main_layout.addLayout(right_layout, stretch=1)

    # -------------------------
    # Loading images & labels
    # -------------------------

    def load_raw_image(self, image_name: str):
        path = self.images_dir / image_name
        if not path.exists():
            raise FileNotFoundError(f"Raw image file not found: {path}")
        img = Image.open(str(path))
        arr = np.array(img)
        return arr

    def load_label_image(self, image_name: str):
        p1 = self.labels_dir / image_name
        stem = Path(image_name).stem
        p2 = self.labels_dir / f"{stem}{self.label_suffix}"

        for p in (p1, p2):
            if p.exists():
                img = Image.open(str(p))
                arr = np.array(img)
                # if bits are stored in multiple channels, take first
                if arr.ndim == 3:
                    arr = arr[..., 0]
                if arr.ndim != 2:
                    raise ValueError(f"Label image must be 2D, got {arr.shape} for {p}")
                return arr.astype(np.int32)

        raise FileNotFoundError(f"No label image found for {image_name} in {self.labels_dir}")

    def load_current_image(self):
        self.current_image_name = self.image_names[self.current_image_idx]
        self.current_indices = self.image_to_indices[self.current_image_name]

        self.image_label.setText(
            f"Image: {self.current_image_name}  "
            f"({self.current_image_idx+1}/{len(self.image_names)})"
        )

        # load raw + labels
        self.current_raw = self.load_raw_image(self.current_image_name)
        self.current_label_img = self.load_label_image(self.current_image_name)

        # make overlay
        overlay = build_overlay_image(self.current_raw, self.current_label_img)
        pixmap = numpy_to_qpixmap(overlay)
        self.image_view.set_pixmap(pixmap)

        # populate list + markers
        self.populate_list_and_markers()

    # -------------------------
    # UI helpers
    # -------------------------

    def populate_list_and_markers(self):
        self.matches_list.clear()
        self.image_view.clear_markers()
        self.current_selected_global_idx = None
        self.global_to_row_index = {}

        rows = self.matches_df.loc[self.current_indices]

        for row_idx, (global_idx, row) in enumerate(rows.iterrows()):
            cell_id = row["cell_id"]
            x = row["x"]
            y = row["y"]
            matched_label = row["matched_label"]

            matched = (matched_label != -1)
            dist = row.get("distance_to_segment", np.nan)

            text = f"cell_id={cell_id} | match={matched_label} | dist={dist:.2f}"
            item = QListWidgetItem(text)
            color = QColor(0, 150, 0) if matched else QColor(200, 0, 0)
            item.setForeground(color)
            item.setData(Qt.ItemDataRole.UserRole, global_idx)  # store global index
            self.matches_list.addItem(item)

            self.global_to_row_index[global_idx] = row_idx

            # add marker (none selected initially)
            self.image_view.add_marker(x, y, global_idx, matched=matched, selected=False)

    def update_list_item_for_row(self, global_idx):
        # find item with given global_idx
        row_idx = self.global_to_row_index.get(global_idx, None)
        if row_idx is None:
            return

        item = self.matches_list.item(row_idx)
        row = self.matches_df.loc[global_idx]
        matched_label = row["matched_label"]
        matched = (matched_label != -1)
        dist = row.get("distance_to_segment", np.nan)
        text = f"cell_id={row['cell_id']} | match={matched_label} | dist={dist:.2f}"
        item.setText(text)
        color = QColor(0, 150, 0) if matched else QColor(200, 0, 0)
        item.setForeground(color)

        selected = (self.current_selected_global_idx == global_idx)
        self.image_view.update_marker_style(global_idx, matched=matched, selected=selected)

    # -------------------------
    # Navigation
    # -------------------------

    def prev_image(self):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_image_idx < len(self.image_names) - 1:
            self.current_image_idx += 1
            self.load_current_image()

    # -------------------------
    # List selection
    # -------------------------

    def on_list_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        # un-highlight the previously selected marker
        if previous is not None:
            prev_idx = previous.data(Qt.ItemDataRole.UserRole)
            if prev_idx is not None:
                row = self.matches_df.loc[prev_idx]
                matched = (row["matched_label"] != -1)
                self.image_view.update_marker_style(prev_idx, matched=matched, selected=False)

        # highlight the newly selected marker
        if current is not None:
            cur_idx = current.data(Qt.ItemDataRole.UserRole)
            if cur_idx is not None:
                self.current_selected_global_idx = cur_idx
                row = self.matches_df.loc[cur_idx]
                matched = (row["matched_label"] != -1)
                self.image_view.update_marker_style(cur_idx, matched=matched, selected=True)
        else:
            # nothing selected
            self.current_selected_global_idx = None

    # -------------------------
    # Marker clicks (from ImageView)
    # -------------------------

    def on_marker_clicked(self, global_idx: int):
        """
        Called when the user left-clicks a marker in the image.
        Selecting same marker again toggles it off (deselect).
        """
        # if already selected, deselect
        if self.current_selected_global_idx == global_idx:
            idx = global_idx
            row = self.matches_df.loc[idx]
            matched = (row["matched_label"] != -1)
            self.image_view.update_marker_style(idx, matched=matched, selected=False)
            self.current_selected_global_idx = None
            self.matches_list.clearSelection()
            return

        # un-highlight previous
        if self.current_selected_global_idx is not None:
            prev = self.current_selected_global_idx
            row_prev = self.matches_df.loc[prev]
            matched_prev = (row_prev["matched_label"] != -1)
            self.image_view.update_marker_style(prev, matched_prev, selected=False)

        # select new
        self.current_selected_global_idx = global_idx

        # select corresponding row in list
        row_idx = self.global_to_row_index.get(global_idx, None)
        if row_idx is not None:
            item = self.matches_list.item(row_idx)
            self.matches_list.setCurrentItem(item)

        # ensure marker is highlighted
        row = self.matches_df.loc[global_idx]
        matched = (row["matched_label"] != -1)
        self.image_view.update_marker_style(global_idx, matched=matched, selected=True)

    # -------------------------
    # Manual edits
    # -------------------------

    def set_no_match_for_selected(self):
        if self.current_selected_global_idx is None:
            return
        idx = self.current_selected_global_idx
        self.matches_df.loc[idx, "matched_label"] = -1
        self.matches_df.loc[idx, "distance_to_segment"] = np.inf
        self.update_list_item_for_row(idx)

    def on_image_right_click(self, x: float, y: float):
        """
        Called when user right-clicks on the image.
        If a click is selected, we assign it to the segment under (x,y).
        """
        if self.current_selected_global_idx is None:
            return

        if self.current_label_img is None:
            return

        ix = int(round(x))
        iy = int(round(y))

        h, w = self.current_label_img.shape
        if not (0 <= ix < w and 0 <= iy < h):
            return

        label_id = int(self.current_label_img[iy, ix])
        if label_id == 0:
            # background, ignore
            return

        # enforce 1:1: if another click in this image already has this label,
        # unset it (set to -1)
        for global_idx in self.current_indices:
            if global_idx == self.current_selected_global_idx:
                continue
            row = self.matches_df.loc[global_idx]
            if row["matched_label"] == label_id:
                self.matches_df.loc[global_idx, "matched_label"] = -1
                self.matches_df.loc[global_idx, "distance_to_segment"] = np.inf
                self.update_list_item_for_row(global_idx)

        # assign label to selected click
        idx = self.current_selected_global_idx
        row = self.matches_df.loc[idx]
        x_pt = row["x"]
        y_pt = row["y"]

        ys, xs = np.where(self.current_label_img == label_id)
        if xs.size > 0:
            coords = np.stack([xs, ys], axis=1).astype(np.float32)
            pt = np.array([x_pt, y_pt], dtype=np.float32)
            d2 = np.sum((coords - pt) ** 2, axis=1)
            dist = float(np.sqrt(d2.min()))
        else:
            dist = np.inf

        self.matches_df.loc[idx, "matched_label"] = label_id
        self.matches_df.loc[idx, "distance_to_segment"] = dist
        self.update_list_item_for_row(idx)

    def save_matches_csv(self):
        self.matches_df.to_csv(self.matches_csv, index=False)
        self.statusBar().showMessage(f"Saved updated matches to {self.matches_csv}", 5000)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="GUI to manually review and correct segmentation matches.")
    ap.add_argument("--images-dir", type=str, required=True,
                    help="Directory with raw images.")
    ap.add_argument("--labels-dir", type=str, required=True,
                    help="Directory with label masks.")
    ap.add_argument("--matches-csv", type=str, required=True,
                    help="Path to matches.csv from evaluation script.")
    ap.add_argument("--label-suffix", type=str, default="",
                    help="Optional suffix for label files, e.g. '_labels.tif'.")
    return ap.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    win = ReviewWindow(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        matches_csv=Path(args.matches_csv),
        label_suffix=args.label_suffix,
    )
    win.resize(1400, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
