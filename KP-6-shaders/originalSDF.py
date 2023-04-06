import time
import taichi as ti
import taichi_glsl as ts

ti.init(arch=ti.gpu)

TAU = 2.0 * 3.14159256
ASPECT_RATIO = 16 / 9
HEIGHT = 600
WIDTH = int(ASPECT_RATIO * HEIGHT)
RESOLUTION = WIDTH, HEIGHT
RESOLUTION_FLOAT = ts.vec2(float(WIDTH), float(HEIGHT))

pixels = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)


@ti.func
def circle_sdf(coords, radius, center):
    """
    :param coords:
        A 2D vector representing the point (x, y) to calculate the signed distance field for.
    :param radius:
        The radius of the circle.
    :param center:
        A 2D vector representing the center (x, y) of the circle.
    :return:
        The signed distance field value for the input point 'coords' and the circle.
    """
    return (coords - center).norm() - radius


@ti.func
def box_sdf(coords, half_extents):
    """
    :param coords:
        A 2D vector representing the point (x, y) to calculate the signed distance field for.
    :param half_extents:
        A 2D vector representing the half-extents (half_width, half_height) of the axis-aligned rectangle.
    :return:
        The signed distance field value for the input point 'coords' and the rectangle.
    """
    d = abs(coords) - half_extents
    return ti.max(d, 0.).norm() + ti.min(ti.max(d[0], d[1]), 0.)


@ti.func
def ease_in_out_cubic(x):
    """
    This function implements an ease-in-out cubic curve, which is a smooth curve that accelerates
    gradually from zero velocity, reaches peak velocity in the middle, and decelerates smoothly
    to zero velocity again. It is used for animations and transitions.

    :param x:
        A float value between 0 and 1, representing the normalized time or progress of the animation or transition.
    :return:
        The ease-in-out cubic interpolated value, which represents the current state or
        position of the animation or transition.
    """
    return 4 * x * x * x if x < 0.5 else 1 - ((-2 * x + 2) ** 3) / 2.


@ti.func
def rotation_matrix(angle):
    """
    This function returns a 2x2 rotation matrix that rotates a 2D vector
    by the specified angle 'a' in radians.

    :param angle:
        The rotation angle in radians.
    :return:
        A 2x2 matrix representing the 2D rotation.
    """
    c = ti.cos(angle)
    s = ti.sin(angle)
    return ts.mat([c, -s], [s, c])


@ti.kernel
def render(time: ti.f32):
    """
    The render function is a Taichi kernel that generates a two-dimensional animation
    of a circular shape morphing into a square shape, with a pulsating effect.
    The function takes as input the current time and the current frame number,
    and updates the pixel values of the output image.
    """

    # rotates the UV coordinates by an angle that depends on the easeInOutCubic function
    rotation = rotation_matrix(ease_in_out_cubic(ts.fract(time)) * TAU)

    for frag_coord in ti.grouped(pixels):
        uv = (frag_coord - 0.5 * RESOLUTION_FLOAT) / RESOLUTION_FLOAT[1]
        uv = rotation @ uv
        color = ts.vec3(0.)

        # calculates a time-dependent weight value used to linearly interpolate
        time_dependent_weight = 0.5 + 0.5 * ti.sin(TAU * time)

        # draw separate lines
        radius = 0.
        while radius <= 0.10:
            radius += 0.02

            # calculates the signed distance field value
            circle_sdf_value = circle_sdf(uv, .25 - radius, ts.vec2(0., 0.))
            box_sdf_value = box_sdf(uv, ts.vec2(.12 - radius, .12 - radius))

            # calculates a linear interpolation between two signed distance field values
            lerp_sdf = ts.mix(circle_sdf_value, box_sdf_value, time_dependent_weight)

            color += ts.smoothstep(abs(lerp_sdf) - .005, 1.0 / RESOLUTION_FLOAT[1], 0.)

        pixels[frag_coord] = ts.clamp(color, 0., 1.)


if __name__ == "__main__":
    gui = ti.GUI("Morphing Shape", res=RESOLUTION, fast_gui=True)
    frame = 0
    start_time = time.time()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break

        current_time = time.time() - start_time
        render(current_time)
        gui.set_image(pixels)
        gui.show()
        frame += 1

    gui.close()