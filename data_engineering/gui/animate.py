import os
import matplotlib.pyplot as plt

class Animate():
    
    def __init__(self, path = None):
        self.path = path
        self.playing = False
    
    def play(self):
        self.generate_canvas()
        while self.playing:
            for img in os.scandir(self.path):
                plt.plot()
        
    def generate_canvas(self):
        
    
    
    def is_playing(self):
        return self.playing
    
            
            
        