from gym.envs.classic_control import rendering
import pyglet

class Text(rendering.Geom):

    def __init__(self, text, size=14):
        rendering.Geom.__init__(self)
        self.size = size
        self.set_text(text)

    def set_text(self, text):
        self.text = pyglet.text.Label(text, 'sans-serif', self.size)
    
    def render1(self):
        self.text.draw()