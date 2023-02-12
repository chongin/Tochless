import tkinter as tk
from PIL import ImageTk, Image
import os
import subprocess

print(os.getcwd())

process = None
instructions = """
Pinch Gesture : 
if pinky finger is outwards it will hold
if pinky finger is inwards it will double click

Release Gesture :\nRelease the Hold Button
"""

for i in os.listdir(os.getcwd()):
	print(i)

def do():
	global process
	process = subprocess.Popen(['python3', 'Start.py'])

def do_game():
	global process
	process = subprocess.Popen(['python3', 'game.py'])

def cl():
	global process
	process.kill()

win = tk.Tk()
win.geometry("400x400")

imgStart = ImageTk.PhotoImage(Image.open("media/Start.png"))
imgStop = ImageTk.PhotoImage(Image.open("media/Stop.png"))

label = tk.Label(win,text='Touchless',font=('Arial',40))
label.pack()

fr = tk.Frame(win)
fr.pack()

buttonStart = tk.Button(fr,image=imgStart,command=do)
buttonStart.pack(side='left')

buttonStop = tk.Button(fr,image=imgStop,command=cl)
buttonStop.pack(side='left')

btngame = tk.Button(win,text="        Game        ",font=('Arial',20), command=do_game)
btngame.pack()

insLabel = tk.Label(win,text=instructions,justify='left')
insLabel.pack()

win.mainloop()