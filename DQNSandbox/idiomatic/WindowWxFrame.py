import wx

class windowClass(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(windowClass, self).__init__(*args, **kwargs)
        print("window class init called ... ")


    def basicGUI(self):
        print("basdicGUI called ... ")

if __name__ == "__main__":
    app = wx.App()
    windowClass(None, title="my title")
    app.MainLoop()