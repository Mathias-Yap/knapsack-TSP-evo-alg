import tkinter as tk
from tkinter import ttk  # Normal Tkinter.* widgets are not themed!
from ttkthemes import ThemedTk

class start_menu:   
    def __init__(self,root):
        self.root = root
        self.root.title("Genetic Algorithm")
        self.root.configure(background = '#FAFBFC') # background='#383838'
        self.set_window_geometry(600, 600)
        self.make_popsize_slider()
        self.set_problem()
        self.make_crosmut_method()

    def set_window_geometry(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def set_problem(self):
        self.problem_var = tk.StringVar(value="traveling_salesman")  # Default value

        problem_frame = ttk.Frame(self.root)
        problem_frame.grid(row = 1,column = 0, sticky = 'ew')

        ttk.Label(problem_frame, text="Problem Type:").grid(row = 0, column = 0)

        ttk.Radiobutton(problem_frame, text="Traveling Salesman", variable=self.problem_var, value="traveling_salesman",takefocus=0).grid(row = 0, column = 1)
        ttk.Radiobutton(problem_frame, text="Knapsack", variable=self.problem_var, value="knapsack",takefocus=0).grid(row = 0, column = 2)

    
    def make_crosmut_method(self):
        self.crosmut_frame = ttk.Frame(self.root)
        self.crosmut_frame.grid(row = 2,column = 0, sticky = 'ew')

        # Crossover Frame and Combobox
        crosframe = ttk.Frame(self.crosmut_frame)
        crosframe.grid(row = 0, column = 0)
        ttk.Label(crosframe, text="Crossover:").pack(side='top')
        self.cross_box = ttk.Combobox(crosframe, state="readonly")
        self.cross_box.pack(side='top')

        # Mutation Frame and Combobox
        mutframe = ttk.Frame(self.crosmut_frame)
        mutframe.grid(row = 0, column = 1)
        ttk.Label(mutframe, text="Mutation:").pack(side='top')
        self.mut_box = ttk.Combobox(mutframe, state="readonly")
        self.mut_box.pack(side='top')

        # Selection Frame and Combobox
        self.selframe = ttk.Frame(self.crosmut_frame)
        self.selframe.grid(row = 0, column = 2)
        ttk.Label(self.selframe, text="Selection:").pack(side='top')
        self.sel_box = ttk.Combobox(self.selframe, state="readonly")
        self.sel_box.pack(side='top')

        self.genframe = ttk.Frame(self.crosmut_frame)
        self.genframe.grid(row = 0, column = 3)
        ttk.Label(self.genframe, text="Genome:").pack(side='top')
        self.gen_box = ttk.Combobox(self.genframe, state="readonly")
        self.gen_box.pack(side='top')
        # Initialize combobox values based on the current problem type
        self.update_crosmut_options()

        # Bind the problem type selection to update the comboboxes when it changes
        self.problem_var.trace_add("write", lambda name, index, mode: self.update_crosmut_options())


    def update_crosmut_options(self):
        problem = self.problem_var.get()
        if problem == "knapsack":
            self.cross_box['values'] = ("uniform", "one point", "two point")
            self.mut_box['values'] = ("bit flip",)
            self.sel_box['values'] = ("tournament", "roulette")
            self.gen_box['values'] = ("bitstring")
        elif problem == "traveling_salesman":
            self.cross_box['values'] = ()
            self.mut_box['values'] = ()
            self.sel_box['values'] = ()

        # Optionally clear the current selection
        self.cross_box.set('')
        self.mut_box.set('')
        self.sel_box.set('')
        
    def make_popsize_slider(self):
        self.popsize_frame =ttk.Frame(self.root)
        label = ttk.Label(self.popsize_frame, text="population size")
        label.grid(row=0,column=0)
        current_value = tk.DoubleVar()
        self.slider_value_label = ttk.Label(self.popsize_frame, text="19")
        self.slider_value_label.grid(row=0,column=2)
        slider = ttk.Scale(self.popsize_frame,length=500, from_=10, to=99, orient="horizontal",name = "population size",variable=current_value,command=self.update_slider_value)
        slider.set(19)
        slider.grid(row=0,column=1)
       
        
        self.popsize_frame.grid(row = 3,column = 0, sticky = 'ew')
        
    def update_slider_value(self,value):
        self.slider_value_label.configure(text=f"{int(float(value))}")
if __name__ == "__main__":
    root =  ThemedTk(theme="adapta")
    start_menu(root)
    root.mainloop()