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
    # Define base colors in the RGB space
    rgb_base = 1.0 / ts.vec3(255.0)
    bg_color = pow(rgb_base * ts.vec3(14.0, 17.0, 23.0), ts.vec3(2.0))
    fg_color = pow(rgb_base * ts.vec3(24.0, 27.0, 34.0), ts.vec3(2.0))
    hi_color = pow(rgb_base * ts.vec3(132.0, 210.0, 91.0), ts.vec3(2.0))
    anti_aliasing = 2.0 / RESOLUTION_F[1]
    cell_aa = CELL_WIDTH
    current_pos = p

    # Calculate the new position
    new_pos, current_pos = mod1(current_pos, ts.vec2(CELL_WIDTH))
    color = bg_color

    if abs(new_pos.y) < 14.0:
        # Animate the new position based on time
        new_pos.x += t * 10.0

        # Calculate the new end position using the mod1 function
        new_end_pos, new_pos.x = mod1(new_pos.x, 24.0)

        # Calculate noise values using hash function
        noise_hash0 = hash(new_end_pos + 123.4)
        noise_hash1 = ts.fract(8667.0 * noise_hash0)

        # Calculate the final position
        end_pos = new_pos * CELL_WIDTH

        # Calculate radius and phase time using noise values
        radius = ts.mix(0.25, 0.5, noise_hash0) * 1.0
        phase_time = ts.mix(0.5, 1.0, noise_hash1)

        # Calculate the fractional time
        fract_time = (t + phase_time * noise_hash1) % phase_time

        # Update the position's Y value
        end_pos.y -= -ts.mix(2.0, 1.0, noise_hash1) * (fract_time - phase_time * 0.5) * (
                    fract_time - phase_time * 0.5) + radius - 0.5 + 0.125

        # Rotate the position
        end_pos = end_pos @ rot(2.0 * ts.mix(0.5, 0.25, noise_hash0) * t + TAU * noise_hash0)

        # Calculate the distance using the hexagram function
        end_dist = hexagram(end_pos, 0.5 * radius)
        end_dist = abs(end_dist) - 0.05

        # Calculate the current distance using the box function
        current_dist = box(current_pos, ts.vec2(BOX_WIDTH - BOX_RADIUS)) - BOX_RADIUS

        # Calculate the color by mixing the foreground and highlight colors based on end distance
        end_color = ts.mix(fg_color, hi_color, ts.smoothstep(end_dist, cell_aa, -cell_aa))

        # Update the final color by mixing the current color and the end color based on current distance
        color = ts.mix(color, end_color, ts.smoothstep(current_dist, anti_aliasing, -anti_aliasing))

    color = ts.sqrt(color)
    return color


@ti.kernel
def render(t: ti.f32, frame: ti.int32):
    for frag_coord in ti.grouped(pixels):
        uv = (frag_coord - 0.5 * RESOLUTION_F) / RESOLUTION_F[1]
        uv *= 2.0

        # Calculate effect based on UV coordinates and time
        col = effect(uv, t)
        pixels[frag_coord] = ts.clamp(col, 0.0, 1.0)


if __name__ == "__main__":
    gui = ti.GUI("original Stars", res=RESOLUTION, fast_gui=True)
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
