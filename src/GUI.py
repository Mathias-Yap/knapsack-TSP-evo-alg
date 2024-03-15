
from tkinter import ttk
import tkinter as tk
from ttkthemes import ThemedTk
class start_menu:   
    def __init__(self,root):
        self.root = root
        self.root.title("Genetic Algorithm")
        self.root.configure()
        self.set_window_geometry(1000, 600)
        self.make_popsize_slider()

    def set_window_geometry(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def make_popsize_slider(self):
        popsize_frame =ttk.Frame(self.root)
        label = ttk.Label(popsize_frame, text="population size")
        label.pack(side = 'left')
        slider = ttk.Scale(popsize_frame,length=600, from_=0, to=200, orient="horizontal",name = "population size")
        slider.set(19)
        slider.pack(side = 'right')
        popsize_frame.pack(side = 'bottom')
if __name__ == "__main__":
    root = tk.Tk()
    start_menu(root)
    root.mainloop()