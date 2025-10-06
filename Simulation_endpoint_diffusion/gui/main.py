import sys
import time
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QStatusBar, QFrame, QSizePolicy, QGridLayout
)
from PyQt6.QtCore import QThread, QObject, pyqtSignal, Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont
import simulation as sim

# A worker class to run the simulation in a separate thread
class SimulationWorker(QObject):
    """
    Runs the simulation in a background thread to prevent the GUI from freezing.
    Emits signals to update the GUI with the simulation's progress.
    Can be paused, resumed, and stopped.
    """
    progress = pyqtSignal(float, np.ndarray)  # Signal to emit current time and particle positions
    finished = pyqtSignal(str)               # Signal to emit a finishing message
    
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self._is_running = True
        self._is_paused = False
        self.speed = self.parameters.get("simulation_speed", 1.0)

    def run(self):
        """
        Starts the simulation loop.
        """
        try:
            rs = np.random.RandomState(seed=self.parameters["seed"])
            cell_speed = np.sqrt(2 * self.parameters["diffusion"])
            arena = sim.RectangularArena(self.parameters["arena_rect"], self.parameters["edge_force"], self.parameters["radius"])
            beta = 0.8
            
            events = [sim.BirthEvent(self.parameters["alpha"], beta, self.parameters["ceta"], self.parameters["radius"])]
            if self.parameters.get("mu", 0) > 0:
                events.append(sim.DeathEvent(self.parameters["mu"]))

            collision_function = sim.createRigidPotential(self.parameters["repulsion_force"], 2 * self.parameters["radius"])

            simulation_instance = sim.Simulation(
                minTimeStep=50,
                initialParticles=self.parameters["num_particles"],
                maxParticles=self.parameters["max_particles"],
                particleSpeed=cell_speed,
                arena=arena,
                particleCollision=collision_function,
                particleCollisionMaxDistance=2 * self.parameters["radius"],
                events=events,
                rs=rs
            )
            
            simulation_generator = simulation_instance.simulate(self.parameters["simulation_length"])

            for t, P in simulation_generator:
                if not self._is_running:
                    break
                
                while self._is_paused:
                    time.sleep(0.1)
                    if not self._is_running:
                        break
                
                if not self._is_running:
                    break

                self.progress.emit(t, P)
                if self.speed > 0:
                    time.sleep(0.01 / self.speed)

            self.finished.emit("Simulation finished or stopped.")

        except Exception as e:
            print(f"Error in simulation thread: {e}")
            self.finished.emit(f"Error: {e}")

    def stop(self):
        self._is_running = False
        self._is_paused = False

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def update_speed(self, speed):
        self.speed = speed

# A custom widget to draw the simulation
class SimulationCanvas(QWidget):
    """
    The canvas where the simulation is drawn. Handles coordinate transformation,
    zooming, and painting of cells and the arena boundary.
    """
    def __init__(self, arena_rect):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.particles = np.array([])
        self.time = 0.0
        self.cell_radius = 7.5 
        self.cell_radius_pixels = 5
        self.arena_rect = arena_rect
        self.setStyleSheet("background-color: #1a1a1a;")
        self.zoom_level = 1.0

    def update_data(self, t, particles, radius):
        self.time = t
        self.particles = particles
        self.cell_radius = radius
        self.update()

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        if angle > 0:
            self.zoom_level *= 1.05     # 1.15 
        else:
            self.zoom_level /= 1.05     # 1.15
        self.zoom_level = max(0.1, min(self.zoom_level, 10.0))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        widget_w, widget_h = self.width(), self.height()
        sim_w = self.arena_rect[2] - self.arena_rect[0]
        sim_h = self.arena_rect[3] - self.arena_rect[1]
        
        scale = min(widget_w / sim_w, widget_h / sim_h) * 0.95 * self.zoom_level
        offset_x = (widget_w - sim_w * scale) / 2
        offset_y = (widget_h - sim_h * scale) / 2
        self.cell_radius_pixels = self.cell_radius * scale
        
        arena_x, arena_y = int(offset_x), int(offset_y)
        arena_w, arena_h = int(sim_w * scale), int(sim_h * scale)

        painter.setPen(QPen(QColor("#404040"), 2))
        painter.drawRect(arena_x, arena_y, arena_w, arena_h)

        if self.particles is not None and len(self.particles) > 0:
            painter.setPen(QPen(QColor("#00aaff"), 1))
            painter.setBrush(QBrush(QColor("#00aaff")))
            for p in self.particles:
                tx = (p[0] - self.arena_rect[0]) * scale + offset_x
                ty = (p[1] - self.arena_rect[1]) * scale + offset_y
                painter.drawEllipse(int(tx - self.cell_radius_pixels), int(ty - self.cell_radius_pixels), int(2 * self.cell_radius_pixels), int(2 * self.cell_radius_pixels))

        painter.setPen(QColor("white"))
        font = QFont()
        font.setPointSize(14)
        painter.setFont(font)

        total_seconds = int(self.time)
        days = total_seconds // (24 * 3600)
        remaining_seconds = total_seconds % (24 * 3600)
        hours = remaining_seconds // 3600
        minutes = (remaining_seconds % 3600) // 60
        
        time_str = f"Time: {days}d {hours:02d}h {minutes:02d}m"
        cell_str = f"Cells: {len(self.particles)}"
        
        painter.drawText(20, 30, time_str)
        painter.drawText(20, 60, cell_str)


# Main Application Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agent-Based Cell Simulation")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        self.arena_side = 351 * 1.3
        self.arena_rect = [-self.arena_side, -self.arena_side, self.arena_side, self.arena_side]
        self.canvas = SimulationCanvas(self.arena_rect)
        
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)
        
        self.param_widgets = {}
        #  Create a grid layout for parameters 
        self.params_grid_layout = QGridLayout()
        self._create_parameter_widgets(self.params_grid_layout)
        controls_layout.addLayout(self.params_grid_layout)

        controls_layout.addStretch(1)

        button_layout = QHBoxLayout()
        self.play_stop_button = QPushButton("Stop")
        self.restart_button = QPushButton("Restart Simulation")
        button_layout.addWidget(self.play_stop_button)
        button_layout.addWidget(self.restart_button)
        controls_layout.addLayout(button_layout)

        self.main_layout.addWidget(self.canvas)
        self.main_layout.addWidget(controls_widget)
        
        self.main_layout.setStretchFactor(self.canvas, 1)
        self.main_layout.setStretchFactor(controls_widget, 0)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.simulation_thread = None
        self.simulation_worker = None
        self.is_paused = False

        self.play_stop_button.clicked.connect(self.toggle_pause)
        self.restart_button.clicked.connect(self.restart_simulation)
        
        self._apply_styles()
        self.restart_simulation()

    def _create_parameter_widgets(self, layout):
        params_config = {
            "num_particles": {"label": "Initial Particles", "type": "int", "min": 1, "max": 1000, "val": 100},
            "max_particles": {"label": "Max Particles", "type": "int", "min": 100, "max": 5000, "val": 2000},
            "simulation_days": {"label": "Sim Length (Days)", "type": "double", "min": 0.1, "max": 9999.0, "val": 4.0, "step": 0.1},
            "simulation_speed": {"label": "Sim Speed (x)", "type": "double", "min": 0.1, "max": 100.0, "val": 1.0, "step": 0.1},
            "log10_alpha": {"label": "Prolif. Rate (log α)", "type": "double", "min": -10.0, "max": 0.0, "val": -4.8, "step": 0.1},
            "log10_mu": {"label": "Death Rate (log μ)", "type": "double", "min": -10.0, "max": 0.0, "val": -8.0, "step": 0.1},
            "diffusion": {"label": "Diffusion (D)", "type": "double", "min": 0.0, "max": 9999.0, "val": 0.5, "step": 0.1},
            "radius": {"label": "Cell Radius (μm)", "type": "double", "min": 1.0, "max": 100.0, "val": 7.5, "step": 0.5},
            "repulsion_force": {"label": "Repulsion Force", "type": "double", "min": 0.0, "max": 100.0, "val": 0.1, "step": 0.01},
            "edge_force": {"label": "Edge Force", "type": "double", "min": 0.0, "max": 100.0, "val": 0.01, "step": 0.01},
            "ceta": {"label": "Inhibition Const (γ)", "type": "double", "min": 0.0, "max": 100.0, "val": 1.0, "step": 0.1},
            "seed": {"label": "Random Seed", "type": "int", "min": 0, "max": 999999, "val": 42},
        }

        #Loop to populate the grid layout in two columns 
        for i, (name, config) in enumerate(params_config.items()):
            row = i // 2
            col = (i % 2) * 2

            label = QLabel(config["label"])
            widget = QSpinBox() if config["type"] == "int" else QDoubleSpinBox()
            if config["type"] == "double": widget.setSingleStep(config.get("step", 0.1))
            
            widget.setRange(config["min"], config["max"])
            widget.setValue(config["val"])
            
            layout.addWidget(label, row, col)
            layout.addWidget(widget, row, col + 1)
            
            self.param_widgets[name] = widget

            if name == "simulation_speed":
                widget.valueChanged.connect(self.update_simulation_speed)

    def _stop_current_simulation(self):
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_worker.progress.disconnect()
            self.simulation_worker.finished.disconnect()
            self.simulation_worker.stop()
            self.simulation_thread.quit()
            self.simulation_thread.wait()
        self.simulation_thread = None
        self.simulation_worker = None

    def restart_simulation(self):
        self._stop_current_simulation()

        params = {name: widget.value() for name, widget in self.param_widgets.items()}
        params["simulation_length"] = params.pop("simulation_days") * 24 * 3600
        params["alpha"] = 10**params.pop("log10_alpha")
        # Get mu value from parameters
        params["mu"] = 10**params.pop("log10_mu")
        params["arena_rect"] = self.arena_rect
        
        self.simulation_thread = QThread()
        self.simulation_worker = SimulationWorker(params)
        self.simulation_worker.moveToThread(self.simulation_thread)

        self.simulation_thread.started.connect(self.simulation_worker.run)
        self.simulation_worker.finished.connect(self.on_simulation_finished)
        self.simulation_worker.progress.connect(self.update_frame)

        self.simulation_thread.start()
        self.play_stop_button.setText("Stop")
        self.play_stop_button.setEnabled(True)
        self.is_paused = False

    def toggle_pause(self):
        if not self.simulation_thread or not self.simulation_thread.isRunning():
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.simulation_worker.pause()
            self.play_stop_button.setText("Play")
            self.status_bar.showMessage("Simulation Paused.")
        else:
            self.simulation_worker.resume()
            self.play_stop_button.setText("Stop")
            self.status_bar.showMessage("Simulation Resumed.")

    def update_simulation_speed(self, speed):
        if self.simulation_worker:
            self.simulation_worker.update_speed(speed)

    def on_simulation_finished(self, message):
        self.status_bar.showMessage(message)
        self.play_stop_button.setText("Play")
        self.play_stop_button.setEnabled(False)

    def update_frame(self, t, particles):
        radius = self.param_widgets["radius"].value()
        self.canvas.update_data(t, particles, radius)
        
        total_seconds = int(t)
        days = total_seconds // (24 * 3600)
        remaining_seconds = total_seconds % (24 * 3600)
        hours = remaining_seconds // 3600
        minutes = (remaining_seconds % 3600) // 60
        time_str = f"Time: {days}d {hours:02d}h {minutes:02d}m"
        cell_str = f"Cells: {len(particles)}"
        self.status_bar.showMessage(f"{time_str} | {cell_str}")
        
    def closeEvent(self, event):
        self._stop_current_simulation()
        event.accept()

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QWidget { color: #dcdcdc; }
            QLabel { font-size: 10pt; }
            QDoubleSpinBox, QSpinBox {
                background-color: #3c3f41; border: 1px solid #555;
                border-radius: 4px; padding: 5px; font-size: 10pt;
            }
            QPushButton {
                background-color: #007acc; border: none; color: white;
                border-radius: 5px; font-size: 11pt; padding: 8px;
            }
            QPushButton:hover { background-color: #008ae6; }
            QPushButton:pressed { background-color: #006bb3; }
            QPushButton:disabled { background-color: #555; }
            QStatusBar { font-size: 10pt; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

