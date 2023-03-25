import ffmpeg
from vispy import app, scene
from vispy.geometry import Rect
import numpy as np
from numba import njit
from funcs2 import *
from datetime import datetime

app.use_app('pyglet')

video = True

N = 5000
dt = 0.04
COEFF = 3

w, h = 800, 700

asp = w / h
perception = 1/35
vrange=(0.05, 0.1)

canvas = scene.SceneCanvas(keys='interactive',
                           bgcolor='black',
                           show=True,
                           size=(w, h))
# создаем камеру
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=(0, 0, asp, 1),
                                  aspect=1)



coeffs1 = np.array([1,  # cohesion
                   0.005,  # alignment
                   0.5,  # separation
                   0.003,  # walls
                   0.1  # noise
                   ])

coeffs2 = np.array([0.5,  # cohesion
                   0.7,  # alignment
                   4,  # separation
                   0.003,  # walls
                    0.2  # noise
                   ])



coeffs3 = np.array([3,  # cohesion
                   6,  # alignment
                   7,  # separation
                   0.003,  # walls
                   0.1  # noise
                   ])
if COEFF == 1:
    coeffs = coeffs1
elif COEFF == 2:
    coeffs = coeffs2
else:
    coeffs = coeffs3


boids = np.zeros((N, 6), dtype=np.float64)
init_boids(boids, asp, vrange=vrange)

arrows = scene.Arrow(arrows=directions(boids,dt),
                     arrow_color='white',
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)

scene.Line(pos=np.array([[0, 0],
                         [asp, 0],
                         [asp, 1],
                         [0, 1],
                         [0, 0]
                         ]),
           color='white',
           connect='strip',
           method='gl',
           parent=view.scene
           )

text = scene.Text('0.0 fps', parent=view.scene, color='red')
text.pos = 0.1, 0.8


if video:
    fname = f"boids_{COEFF}_{N}_{datetime.now().strftime('%H%M%S')}.mp4"
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{w}x{h}", r=60)
            .output(fname, pix_fmt='yuv420p', preset='slower', r=60)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

dzeros = np.zeros((N, N), dtype=float)

def update(event):
    global process


    flocking(boids, perception, coeffs, asp,dzeros)
    propagate(boids, dt, vrange)
    #periodic_walls(boids, asp)

    bounce_walls(boids,asp)

    arrows.set_data(arrows=directions(boids,dt))
    text.text = 'N = {}\ndt = {}\na = {}\nb = {}\nc = {}\nd = {}\ne = {}\n FPS = {}'.format(N,dt, *coeffs, round(canvas.fps, 2))

    if video:
        frame = canvas.render(alpha=False)
        process.stdin.write(frame.tobytes())
        if event.count > 2700:
            app.quit()
            exit()
    else:
        canvas.update(event)


timer = app.Timer(interval=0,
                  start=True,
                  connect=update)

if __name__ == '__main__':
    canvas.measure_fps()
    app.run()
    if video:
        process.stdin.close()
        process.wait()