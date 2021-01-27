from tkinter import Tk, Label, Button

class TestGui:

    def __init__(self, master):
        self.master = master
        master.title = "Title of GUI"

        self.label = Label(master=master, text="some label")
        self.label.pack()

        self.greet_button = Button(master=master, text="Greet button", command=self.greet)
        self.greet_button.pack()

    def greet(self):
        print("Greetings")




root = Tk()
myGui = TestGui(root)
root.mainloop()
