"""
    В этом файле все комментарии обозначают внесенные изменения
"""
import taichi as ti
import taichi_glsl as ts
import time

ti.init(arch=ti.gpu)

PI = 3.141592654
TAU = 2.0 * PI
ASPECT_RATIO = 16 / 9
HEIGHT = 600
WIDTH = int(ASPECT_RATIO * HEIGHT)
RESOLUTION = WIDTH, HEIGHT
RESOLUTION_F = ts.vec2(float(WIDTH), float(HEIGHT))
LAYERS = 5

pixels = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)

CELL_WIDTH = 0.05
BOX_WIDTH = 0.5 * CELL_WIDTH * 23.0 / 32.0
BOX_RADIUS = CELL_WIDTH * 3.0 / 32.0


@ti.func
def mod1(p, size):
    """
    Calculate the integer and fractional part of a 2D point, given a specific cell size.
    :param p:
        The input 2D point.
    :param size:
        The size of the cells.
    :return:
        A tuple containing the integer coordinates and the fractional coordinates
    """
    half_size = size * 0.5
    c = ts.floor((p + half_size) / size)
    p = (p + half_size) % size - half_size
    return c, p


@ti.func
def box(p, b):
    """
    Calculate the signed distance from a 2D point to a rectangle centered at the origin.
    :param p:
        A 2D vector representing the point (x, y) to calculate the signed distance field for.
    :param b:
        A 2D vector representing the half-extents (half_width, half_height) of the axis-aligned rectangle.
    :return:
        The signed distance field value for the input point 'coords' and the rectangle.
    """
    d = abs(p) - b
    return max(d, 0.0).norm() + min(max(d[0], d[1]), 0.0)


@ti.func
def hash(co):
    """
    Calculate a simple hash value for a given 2D point.
    :param co:
        The input 2D point.
    :return:
        The hash value for the input point, in the range [0, 1).
    """
    return ts.fract(ti.sin(co * 12.9898) * 13758.5453)


@ti.func
def dot(a, b):
    """
    Calculate the dot product of two 2D vectors.
    :param a:
        The first input 2D vector.
    :param b:
        The second input 2D vector.
    :return:
        The dot product of the input vectors.
    """
    return a[0] * b[0] + a[1] * b[1]


@ti.func
def rot(a):
    """
    This function returns a 2x2 rotation matrix that rotates a 2D vector
    by the specified angle 'a' in radians.

    :param a:
        The rotation angle in radians.
    :return:
        A 2x2 matrix representing the 2D rotation.
    """
    c = ti.cos(a)
    s = ti.sin(a)
    return ts.mat([c, -s], [s, c])


@ti.func
def hexagram(p, r):
    """
    Calculate the signed distance from a 2D point to a hexagram centered at the origin.
    :param p:
        The input 2D point.
    :param r:
        The radius of the hexagram.
    :return:
        The signed distance from the point to the hexagram.
    """
    k = ts.vec4(-0.5, 0.8660254038, 0.5773502692, 1.7320508076)
    p = abs(p)
    p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy
    p -= 2.0 * min(dot(k.yx, p), 0.0) * k.yx
    p -= ts.vec2(ts.clamp(p.x, r * k.z, r * k.w), r)
    return p.norm() * ts.sign(p.y)

@ti.func
def stars_effect(p, t, flag):
    """
    Calculate the stars effect value for a given 2D point and time. The effect value is used to determine
    the color of the pixel at the input point based on the hexagram pattern and movement of the stars.
    :param p:
        The input 2D point.
    :param t:
        The current time.
    :param flag:
        A boolean value that determines the direction of the stars' movement.
        If True, the stars move in a positive direction, otherwise, they move in a negative direction.
    :return:
        The stars effect value for the given input point and time.
    """
    # choose the direction
    direction = 1 if flag else -1

    current_pos = p
    new_pos, current_pos = mod1(current_pos, ts.vec2(CELL_WIDTH))
    new_pos.x += direction * t * 10.0
    new_end_pos, new_pos.x = mod1(new_pos.x, 24.0)
    noise_hash0 = hash(new_end_pos + 123.4)
    noise_hash1 = ts.fract(8667.0 * noise_hash0)
    end_pos = new_pos * CELL_WIDTH
    radius = ts.mix(0.25, 0.5, noise_hash0)
    phase_time = ts.mix(0.5, 1.0, noise_hash1)
    fract_time = (t + phase_time * noise_hash1) % phase_time
    end_pos.y -= -ts.mix(2.0, 1.0, noise_hash1) * (fract_time - phase_time * 0.5) * (
                fract_time - phase_time * 0.5) + radius - 0.5 + 0.125
    end_pos = end_pos @ rot(2.0 * ts.mix(0.5, 0.25, noise_hash0) * direction * t + TAU * noise_hash0)
    end_dist = hexagram(end_pos, 0.5 * radius)
    end_dist = abs(end_dist) - 0.05

    # calculate the noise, that imposes a "drunk" effect on hexagram
    d = 0.2 * ti.sin(2. * t + 10. * ti.abs((p.x + 1) * (p.y + 1)))

    # There are calculate the interpolation value for ts.mix function in 'effect' func,
    # instead of mixing color to the background
    end_color_pos = ts.smoothstep(end_dist - d * 0.15, CELL_WIDTH, -CELL_WIDTH)

    return end_color_pos

@ti.func
def effect(p, t):
    """
    Calculate the color of a pixel based on the hexagram effect.
    :param p:
        The input 2D point.
    :param t:
        The current time.
    :return:
        The color of the pixel at the input point based on the hexagram effect.
    """
    rgb_base = 1.0 / ts.vec3(255.0)
    bg_color = pow(rgb_base * ts.vec3(255.0, 0.0, 0.0), ts.vec3(2.0))
    fg_color = pow(rgb_base * ts.vec3(24.0, 27.0, 34.0), ts.vec3(2.0))
    hi_color = pow(rgb_base * ts.vec3(255.0, 215.0, 0.0), ts.vec3(2.0))
    hi_color2 = pow(rgb_base * ts.vec3(187.0, 187.0, 187.0), ts.vec3(2.0))
    anti_aliasing = 2.0 / RESOLUTION_F[1]
    current_pos = p
    new_pos, current_pos = mod1(current_pos, ts.vec2(CELL_WIDTH))
    color = bg_color

    # calculate noise for changing background color
    D = 0.2 * ti.sin(2. * t + 1. * ti.abs((-p.x + 2) * (p.y + 2)))
    D1 = 0.5 * ti.sin(2. * t + 1. * ti.abs((p.x + 2) * (p.y + 2)))

    current_dist = box(current_pos, ts.vec2(BOX_WIDTH - BOX_RADIUS)) - BOX_RADIUS

    # there is was changed order of drawing pixels:
    # 1) set background as bg_color
    # 2) mix to the background colV with D1 noice
    # 3) mix to the background boxes with fg_color, which size changing by D noise
    colV = hi_color2
    color = ts.mix(color, colV, ts.smoothstep(D1 * 0.027, anti_aliasing, -anti_aliasing))
    color = ts.mix(color, fg_color, ts.smoothstep(current_dist - D * 0.027, anti_aliasing, -anti_aliasing))

    # get smoothstep value from stars_effect for every line of stars with offset
    end_color_pos1 = stars_effect(p - 2, t, False)
    end_color_pos2 = stars_effect(p - 1, t, True)
    end_color_pos3 = stars_effect(p, t, False)
    end_color_pos4 = stars_effect(p + 1, t, True)
    end_color_pos5 = stars_effect(p + 2, t, False)

    # mix stars color to the background with individual interpolation value.
    # thus the new line of stars does not affect the previous ones with its color
    color = ts.mix(color, hi_color, end_color_pos1)
    color = ts.mix(color, hi_color, end_color_pos2)
    color = ts.mix(color, hi_color, end_color_pos3)
    color = ts.mix(color, hi_color, end_color_pos4)
    color = ts.mix(color, hi_color, end_color_pos5)

    color = ts.sqrt(color)
    return color


@ti.kernel
def render(t: ti.f32, frame: ti.int32):

    # rotates the UV coordinates by sin func
    m = rot(ti.sin(t * 0.5) * 0.5)

    for frag_coord in ti.grouped(pixels):
        uv = (frag_coord - 0.5 * RESOLUTION_F) / RESOLUTION_F[1]

        # displacement of space in a circle in time
        uv.x += ti.sin(t * 0.5) * 0.5
        uv.y += ti.cos(t * 0.5) * 0.5

        uv = 2 * m @ uv
        col = effect(uv, t)
        pixels[frag_coord] = ts.clamp(col, 0.0, 1.0)


if __name__ == "__main__":
    gui = ti.GUI("new Stars", res=RESOLUTION, fast_gui=True)
    frame = 0
    start = time.time()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break

        t = time.time() - start
        render(t, frame)
        gui.set_image(pixels)
        gui.show()
        frame += 1

    gui.close()
