# interface.py
import tkinter as tk
from tkinter import ttk

class EVAInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EVA")
        self.root.configure(bg="black")
        self.root.geometry("1600x900")
        self.root.resizable(False, False)

        self.label = ttk.Label(
            self.root, 
            text="EVA", 
            font=("Helvetica", 36, "bold"),
            foreground="cyan", 
            background="black"
        )
        self.label.place(relx=0.5, rely=0.5, anchor="center")

        self.alpha = 1.0
        self.dim = False
        self.animar()

    def animar(self):
        # Anima levemente o brilho do nome EVA
        if self.dim:
            self.alpha += 0.01
            if self.alpha >= 1.0:
                self.alpha = 1.0
                self.dim = False
        else:
            self.alpha -= 0.01
            if self.alpha <= 0.7:
                self.alpha = 0.7
                self.dim = True
        self.root.attributes("-alpha", self.alpha)
        self.root.after(50, self.animar)

    def iniciar(self):
        self.root.mainloop()
