import time
import numpy as np
import taichi as ti
from numba import njit, prange

ti.init(arch=ti.gpu)

n = 720
ASPECT_RATIO = 16/9
nx, ny = int(ASPECT_RATIO * n), n
RESOLUTION = ny, nx

pixels = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)
past_u = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)
current_u = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)
future_u = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)
accum_u = ti.Vector.field(3, dtype=ti.f32, shape=(RESOLUTION[1],RESOLUTION[0]))
kappa_field = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)

h = 1.0  # пространственный шаг решетки
c = 1.0  # скорость распространения волн
dt = h / (c * 1.5)  # временной шаг

acc = 0.04         # вес кадра для аккумулятора

imp_freq = 400    # "частота" для генерации нескольких волн импульса
imp_sigma = np.array([0.01, 0.03])
s_pos = np.array([-0.5, 0.])  # положение источника
# s_alpha = -np.radians(-20.0)     # направление источника
s_alpha = -np.radians(0.0)
s_rot = np.array([
    [np.cos(s_alpha), -np.sin(s_alpha)],
    [np.sin(s_alpha), np.cos(s_alpha)]
])

hyperbola_s = 1
hyperbola_pos = np.array([0.1, 0.])
hyperbola_s2 = 1
hyperbola_pos2 = np.array([-0.1, 0.])

n = np.array([  # коэффициент преломления
    1.20, # R
    1.35, # G
    1.50  # B
])


@njit
def mix(a, b, x):
    """
        Performs linear interpolation between two values.

        Args:
            a: The start value.
            b: The end value.
            x: The interpolation factor.

        Returns:
            The interpolated value.
    """
    return a * x + b * (1.0 - x)


@njit
def clamp(x, low, high):
    """
        Clamps a value within a specified range.

        Args:
            x: The value to be clamped.
            low: The lower bound of the range.
            high: The upper bound of the range.

        Returns:
            The clamped value.
    """
    return np.maximum(np.minimum(x, high), low)


@njit
def smoothstep(edge0, edge1, x):
    """
        Performs smooth interpolation between two edge values based on a given input.

        Args:
            edge0: The lower edge value.
            edge1: The upper edge value.
            x: The input value.

        Returns:
            The smoothed interpolation value between the edges.
    """
    n = (x - edge0) / (edge1 - edge0)
    t = clamp(n, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit
def wave_impulse(point: np.ndarray,
                 pos: np.ndarray,
                 freq: float,
                 sigma: np.ndarray,
                 ) -> float:
    """
        Calculates the value of a wave impulse at a given point.

        Args:
            point (np.ndarray): 1D numpy array representing the coordinates of the point.
            pos (np.ndarray): 1D numpy array representing the position of the impulse source.
            freq (float): Frequency of the wave impulse.
            sigma (np.ndarray): 1D numpy array representing the sigma values for the wave impulse.

        Returns:
            float: Value of the wave impulse at the given point.
    """
    d = (point - pos) / sigma
    return np.exp(-0.5 * d @ d) * np.cos(freq * point[0])


@njit
def impulse() -> np.ndarray:
    """
        Generates an impulse signal as a 2D numpy array.

        Returns:
            np.ndarray: 2D numpy array representing the impulse signal.
    """
    res = np.zeros((ny, nx), dtype=np.float32)
    for i in prange(1, ny - 1):
        for j in prange(1, nx - 1):
            uv = (np.array([j, i]) - 0.5 * np.array([nx, ny])) / ny
            res[i, j] += wave_impulse(s_rot @ uv, s_pos, imp_freq, imp_sigma)
    return res


@njit
def sd_hyperbola(p, k) -> int:
    """
        Calculates the signed distance to a hyperbola shape and returns an indicator value

        Args:
            p (np.ndarray): Input point coordinates as a 2D numpy array.
            k (float): Constant parameter.

        Returns:
            int: The signed distance to the hyperbola shape (-1 if inside, 1 if outside).
    """
    x = p[0]
    y = p[1]
    p = np.abs(p)

    p = np.array([p[0] - p[1], p[0] + p[1]]) / (2.0) ** (0.5)

    if p[0] * p[1] <= k-0.05 and 0.15 < abs(x) and abs(y) < 0.35 and x > 0:
        return -1
    else:
        return 1


@njit
def sd_hyperbola2(p, k) -> int:
    """
        Calculates the signed distance to a hyperbola shape and returns an indicator value

        Args:
            p (np.ndarray): Input point coordinates as a 2D numpy array.
            k (float): Constant parameter.

        Returns:
            int: The signed distance to the hyperbola shape (-1 if inside, 1 if outside).
    """
    x = p[0]
    y = p[1]
    p = np.abs(p)

    p = np.array([p[0] - p[1], p[0] + p[1]]) / (2.0) ** (0.5)

    if p[0] * p[1] <= k-0.05 and 0.15 < abs(x) and abs(y) < 0.35 and x < 0:
        return -1
    else:
        return 1


@njit
def Hyperbola_mask(a: float = 0.01, b: float = 0.0) -> np.ndarray:
    """
        Generates a mask representing a hyperbolic shape.

        Args:
            a (float, optional): Smoothstep lower edge. Defaults to 0.01.
            b (float, optional): Smoothstep upper edge. Defaults to 0.0.

        Returns:
            np.ndarray: 2D numpy array representing the hyperbola mask.
    """
    res = np.empty((ny, nx), dtype=np.float32)
    for i in prange(ny):
        for j in prange(nx):
            uv = (np.array([j, i]) - 0.5 * np.array([nx, ny])) / ny
            d1 = sd_hyperbola((uv + hyperbola_pos) * hyperbola_s, 0.075)
            d2 = sd_hyperbola2((uv + hyperbola_pos2) * hyperbola_s2, 0.075)
            d = min(d1, d2)
            res[i, j] = smoothstep(a, b, d)
    return res


@ti.kernel
def render():
    """
        Performs calculations on the arrays current_u, future_u, past_u, kappa_field, and accum_u.

        The function updates the values in future_u based on the values in current_u, past_u, and kappa_field.
        It also updates the values in accum_u based on the absolute values of current_u, kappa_field, and some constants
    """
    for y in ti.ndrange(RESOLUTION[1]):
        future_u[0, y] = (current_u[1, y]
                          + (kappa_field[0, y] - 1) / (kappa_field[0, y] + 1)
                          * (future_u[1, y]
                             - current_u[0, y])
                          )
        future_u[RESOLUTION[0]-1, y] = (current_u[RESOLUTION[0]-2, y]
                                        + (kappa_field[RESOLUTION[0]-1, y] - 1) / (kappa_field[RESOLUTION[0]-1, y] + 1)
                                        * (future_u[RESOLUTION[0]-2, y]
                                           - current_u[RESOLUTION[0]-1, y])
                                        )

    for x in ti.ndrange(RESOLUTION[0]):
        future_u[x, 0] = (current_u[x, 1]
                          + (kappa_field[x, 0] - 1) / (kappa_field[x, 0] + 1)
                          * (future_u[x, 1]
                             - current_u[x, 0])
                          )
        future_u[x, RESOLUTION[1]-1] = (current_u[x, RESOLUTION[1]-2]
                                       + (kappa_field[x, RESOLUTION[1]-1] - 1) / (kappa_field[x, RESOLUTION[1]-1] + 1)
                                       * (future_u[x, RESOLUTION[1]-2]
                                          - current_u[x, RESOLUTION[1]-1])
                                       )

    for x in ti.ndrange(RESOLUTION[0]):
        for y in ti.ndrange(RESOLUTION[1]):
            past_u[x, y] = current_u[x, y]
            current_u[x, y] = future_u[x, y]

    for x in ti.ndrange(1, RESOLUTION[0]-1):
        for y in ti.ndrange(1, RESOLUTION[1]-1):
            future_u[x, y] = (
                    kappa_field[x, y] ** 2 * (
                        current_u[x - 1, y] +
                        current_u[x + 1, y] +
                        current_u[x, y - 1] +
                        current_u[x, y + 1] -
                        4 * current_u[x, y])
                    + 2 * current_u[x, y] - past_u[x, y]
            )

    for x in ti.ndrange(RESOLUTION[0]):
        for y in ti.ndrange(RESOLUTION[1]):
            accum_u[y, x] += acc * ti.abs(current_u[x, y]) * kappa_field[x, y]/(c * dt / h)


tmask = Hyperbola_mask(0.01, 0.)

kappa_field.from_numpy((c * dt / h) * (tmask[..., None] / n[None, None, ...] + (1.0 - tmask[..., None])).astype(np.float32))

temp_impuls = np.zeros((RESOLUTION[0], RESOLUTION[1], 3))
temp_impuls = temp_impuls + impulse()[..., None]

past_u.from_numpy(temp_impuls.astype(np.float32))
current_u.copy_from(past_u)
future_u.copy_from(past_u)

if __name__ == "__main__":
    gui = ti.GUI("concave hyperbola and plane", res=(RESOLUTION[1], RESOLUTION[0]), fast_gui=True)
    print(f'window shape {RESOLUTION}')
    frame = 0
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break

        render()
        gui.set_image(accum_u)
        gui.show()
        frame += 1

    gui.close()
