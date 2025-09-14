from PyQt6.QtWidgets import (
    QDialog, QWidget, QFileDialog
)
from typing import Optional
import os

class SelectSeriesDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None, *, defaults: Optional[dict] = None):
        super().__init__(parent)

        CURRENT_PATH = os.getenv(os.curdir)
        
        self.dir_path = QFileDialog.getExistingDirectory(
            parent=self,
            caption="Select directory",
            directory=CURRENT_PATH,
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        
        
    def get_values(self):
        return self.dir_path + "/"
        
        
        
        
        
        
    
