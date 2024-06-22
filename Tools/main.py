import os.path
from moviepy.editor import ImageSequenceClip


class Recorder:
    def __init__(self):
        self.frames: list = []

    def compile(self, destination, fps):
        dir_, _ = os.path.split(destination)
        os.makedirs(dir_, fps, exist_ok=True)
        clip = ImageSequenceClip(self.frames, fps=fps)
        clip.write_videofile(destination, codec='libx264', logger=None)

    def clear(self):
        self.frames.clear()

    def add(self, frame):
        self.frames.append(frame)


