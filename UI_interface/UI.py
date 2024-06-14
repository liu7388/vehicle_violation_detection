import os
import subprocess
import threading
import traceback

import cv2
import wx


class VideoProcessingApp(wx.Frame):
    """
    A GUI application for Vehicle Violation Detection using wxPython.
    """

    def __init__(self):
        """
        Initialize the application window and components.
        """
        super().__init__(parent=None, title="Vehicle Violation Detection", size=(800, 600))
        panel = wx.Panel(self)
        panel.SetBackgroundColour('#003060')

        self.file_path = ""
        self.predict_script_path = "../YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect/predict.py"
        self.second_script_path = "../vehicle/vehicle_detection_video_deepsort.py"

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        title_label = wx.StaticText(panel, label="Vehicle Violation Detection", style=wx.ALIGN_CENTER)
        title_label.SetFont(wx.Font(wx.FontInfo(30).Bold()))
        main_sizer.Add(title_label, 0, wx.ALL | wx.EXPAND, 40)

        # Row 1: Upload Button and Uploaded File Label
        upload_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.upload_button = wx.Button(panel, label="Upload Video")
        self.upload_button.Bind(wx.EVT_BUTTON, self.upload_video)
        upload_sizer.Add(self.upload_button, 0, wx.ALIGN_LEFT | wx.ALL, 20)  # Align left with Process Button

        self.uploaded_file_label = wx.StaticText(panel, label="")
        upload_sizer.Add(self.uploaded_file_label, 0, wx.ALIGN_LEFT | wx.ALL, 20)

        main_sizer.Add(upload_sizer, 0, wx.EXPAND)

        # Row 2: Target ID Label and TextCtrl
        target_id_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Add a spacer to move the Target ID Label slightly to the right
        target_id_sizer.Add(wx.StaticText(panel, label=""), 0, wx.EXPAND | wx.ALL, 10)

        self.target_id_label = wx.StaticText(panel, label="Target ID:")
        target_id_sizer.Add(self.target_id_label, 0, wx.ALIGN_LEFT | wx.ALL, 20)

        self.target_id_text = wx.TextCtrl(panel)
        target_id_sizer.Add(self.target_id_text, 1, wx.ALL | wx.EXPAND, 15)

        main_sizer.Add(target_id_sizer, 0, wx.EXPAND)

        self.process_button = wx.Button(panel, label="Process Video")
        self.process_button.Bind(wx.EVT_BUTTON, self.process_video)
        self.process_button.Disable()
        main_sizer.Add(self.process_button, 0, wx.ALL, 20)

        self.output_label = wx.StaticText(panel, label="", style=wx.ALIGN_LEFT)
        self.output_label.Disable()
        main_sizer.Add(self.output_label, 0, wx.ALL, 20)

        self.processing_label = wx.StaticText(panel, label="")
        self.processing_label.SetFont(wx.Font(wx.FontInfo(15)))
        main_sizer.Add(self.processing_label, 0, wx.ALL, 30)

        panel.SetSizer(main_sizer)

    def upload_video(self, event):
        """
        Opens a file dialog to upload a video file.
        """
        with wx.FileDialog(self, "Choose a file", wildcard="Video Files (*.mp4;*.avi;*.mov)|*.mp4;*.avi;*.mov",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            self.file_path = file_dialog.GetPath()
            self.uploaded_file_label.SetLabel(f"Uploaded file: {os.path.basename(self.file_path)}")
            self.process_button.Enable()

    def process_video(self, event):
        """
        Initiates video processing by starting a separate thread.
        """
        print("Processing...")
        self.processing_label.SetLabel("Processing...")
        target_id = self.target_id_text.GetValue()
        threading.Thread(target=self.run_script, args=(target_id,)).start()

    def run_script(self, target_id):
        """
        Runs the prediction script with subprocess to process the uploaded video.
        """
        try:
            video_name = os.path.basename(self.file_path)
            print(video_name)
            command = f"python {self.second_script_path} --video_name {video_name} --target_id {str(target_id)}"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                result = stdout.decode('utf-8').strip()
                self.run_second_script(target_id=target_id)
            else:
                error = stderr.decode('utf-8').strip()
                print(error)
        except Exception as e:
            traceback.print_exc()
            wx.CallAfter(self.output_label.SetLabel, f"Error: {str(e)}")

    def run_second_script(self, target_id):
        """
        Runs the second script to display the annotated video after processing.
        """
        try:
            video_name = os.path.basename(self.file_path)
            print(video_name)
            command = (f"python ../violation_determination/tools/demo.py"
                       f" --source ../vehicle/data/output/{video_name[:-4]}_annotated.mp4 "
                       f" --target_id {str(target_id)}")

            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                result = stdout.decode('utf-8').strip()
                self.show_result()
            else:
                error = stderr.decode('utf-8').strip()
                print(error)
        except Exception as e:
            traceback.print_exc()
            wx.CallAfter(self.output_label.SetLabel, f"Error: {str(e)}")

    def show_result(self):
        """
        Shows a message box indicating that video processing is completed.
        """
        wx.CallAfter(self.processing_label.SetLabel, "Completed")
        wx.MessageBox("Processing finished! Result will be shown here.", "Result", wx.OK | wx.ICON_INFORMATION)

        video_name = os.path.basename(self.file_path)
        output_video_path = f"./inference/output/{video_name[:-4]}_annotated.mp4"
        self.display_video(output_video_path)

    def display_video(self, video_path):
        """
        Displays the annotated video in a new window.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FPS, 15)

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        wx.CallAfter(self.create_and_show_video_frame, cap, width, height)

    def create_and_show_video_frame(self, cap, width, height):
        """
        Creates and shows a video frame using wxPython.
        """
        new_frame = VideoFrame(cap, width, height)
        new_frame.Show()


class VideoPanel(wx.Panel):
    """
    Panel for displaying video frames using wxPython.
    """

    def __init__(self, parent, width, height):
        """
        Initialize the panel with specified dimensions.
        """
        super().__init__(parent)
        self.SetInitialSize((width, height))
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.frame = None

    def on_paint(self, event):
        """
        Paint event handler to draw the video frame on the panel.
        """
        dc = wx.BufferedPaintDC(self)
        if self.frame is not None:
            h, w = self.frame.shape[:2]
            self.SetSize((w, h))
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            bitmap = wx.Bitmap.FromBuffer(w, h, frame)
            dc.DrawBitmap(bitmap, 0, 0)

    def update_frame(self, frame):
        """
        Updates the current frame to be displayed.
        """
        self.frame = frame
        self.Refresh()


class VideoFrame(wx.Frame):
    """
    Frame for displaying the video using wxPython.
    """

    def __init__(self, cap, width, height):
        """
        Initialize the frame with the video capture and dimensions.
        """
        super().__init__(None, title="Video Player")
        self.panel = VideoPanel(self, width, height)
        self.playing = False
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)

        self.play_button = wx.Button(self, label="Play")
        self.play_button.Bind(wx.EVT_BUTTON, self.on_play)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.panel, 1, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.play_button, 0, wx.CENTER | wx.ALL, 5)

        self.SetSizerAndFit(self.sizer)
        self.Centre()

        self.cap = cap
        self.frame = None

    def on_play(self, event):
        """
        Plays or pauses the video based on current state.
        """
        if not self.playing:
            self.play_button.SetLabel("Pause")
            self.playing = True
            self.timer.Start(30)

    def on_timer(self, event):
        """
        Timer event to update the video frame display.
        """
        ret, self.frame = self.cap.read()
        if ret:
            self.panel.update_frame(self.frame)
        else:
            self.timer.Stop()
            self.play_button.SetLabel("Replay")
            self.playing = False
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


if __name__ == "__main__":
    app = wx.App()
    frame = VideoProcessingApp()
    frame.Show()
    app.MainLoop()
