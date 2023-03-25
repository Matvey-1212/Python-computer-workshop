import numpy as np
from numba import njit, prange


def init_boids(boids: np.ndarray, asp: float, vrange: tuple = (0.01, 1.)):
    '''
        Задает значения массива размерности [n,6] boids: boids[:, :2] - координаты агента
                                                           boids[:, :4] - вектор скорости агента
                                                           boids[:, :6] - вектор ускорения агента

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]
                asp (float): соотношение сторон окна
                vrange (tuple): кортеж, содержащий крайние значения скорости
    '''
    n = boids.shape[0]
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2*np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s

@njit(parallel=True)
def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    '''
        Задает направление движения агента по вектору скорости

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]

                dt (float): временной шаг
        Возвращаемое значение:
                ndarray : нумпай массив направлений движения
    '''
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))

@njit()
def clip_mag(arr: np.ndarray, lims = (0.01, 1.)):
    '''
        Ограничивает скорости всех агентов

        Параметры:
                arr (numpy.ndarray): нумпай массив, содержащий скорости всех агентов
                lims (tuple): кортеж, содержащий крайние значения скорости

    '''

    v = (arr[:, 0]**2 + arr[:, 1]**2)**0.5  #Был изменен способ поска нормы для работы numba

    v_clip = np.clip(v, *lims)
    arr *= (v_clip / v).reshape(-1, 1)




@njit(parallel=True)
def propagate(boids: np.ndarray, dt: float, vrange: tuple):
    '''
        Изменяет положения агентов относительно вектора скорости ивременного шага

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]
                dt (float): временной шаг
                vrange (tuple): кортеж, содержащий крайние значения скорости

    '''
    boids[:, 2:4] += dt * boids[:, 4:6]
    clip_mag(boids[:, 2:4], lims=vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]


@njit()
def periodic_walls(boids: np.ndarray, asp: float):
    '''
        Создает периодичные стены. Если агент выходит за прделы окна, он оявляется с противоположной стороны

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]
                asp (float): соотношение сторон окна

    '''

    for i in prange(boids.shape[0]):  #векторные операции были заменены на циклы для работы numba
        boids[i][0] %= asp
        boids[i][0] %= 1


@njit(parallel=True)
def bounce_walls(boids: np.ndarray, asp: float):
    '''
        Позволяет агентам отскакивать от стен. При попадании на границу
        скорость агента по оси перпендикулярной стене меняется на противоположную по знаку

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]
                asp (float): соотношение сторон окна
    '''

    for i in prange(boids.shape[0]):
        if boids[i][1] <= 0 or boids[i][1] >= 1:
            boids[i][3] = -boids[i][3]
        if boids[i][0] <= 0 or boids[i][0] >= asp:
            boids[i][2] = -boids[i][2]






@njit()
def walls(boids: np.ndarray, asp: float) -> np.ndarray:
    '''
        Возвращает массив из двух элементов, на которые нужно изменить ускорение агентов.
        Коэфиценты расчитываются относительно расстояния до стен.

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]
                asp (float): соотношение сторон окна
        Возвращаемое значение:
                ndarray : нумпай массив направлений движения
    '''

    c = 1e-5
    left = np.abs(boids[:, 0]) + c
    right = np.abs(asp - boids[:, 0]) + c
    bottom = np.abs(boids[:, 1]) + c
    top = np.abs(1 - boids[:, 1]) + c

    ax = 1. / left - 1. / right  #был убран квадрат в знаменателе
    ay = 1. / bottom - 1. / top

    return np.column_stack((ax, ay))



@njit()
def distance(boids: np.ndarray, dzeros) -> np.ndarray:
    '''
        Возвращает энергию системы двумерной решетки при нулевом внешнем магнитном.

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]
                dzeros (numpy.ndarray): массив размерноси [n,n]
        Возвращаемое значение:
                d (numpy.ndarray): матрицa n*n, хранящая расстояния между агентами
    '''

    N = boids.shape[0]
    d = dzeros

    for i in prange(N):  #векторные операции были заменены на циклы
        for j in prange(N):
            d[i][j] = ((boids[i][0] - boids[j][0])**2 + (boids[i][1] - boids[j][1])**2)**0.5


    return d


@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray) -> np.ndarray:
    '''
        Функция взаимодействия агентов по правилу сближения

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]
                idx (int): индекс агента
                neigh_mask (np.ndarray): маска агентов, с которыми будет взаимодействовать изменяемый

        Возвращаемое значение:
                a (numpy.ndarray): массив из двух коэффицентов, на которые нужно изменить ускорения агентов
    '''

    x_median = np.median(boids[neigh_mask, 0]) #поиск среднего был заменен на медиану
    y_median = np.median(boids[neigh_mask, 1])

    a = (np.array([x_median, y_median]) - boids[idx, :2])
    return a


@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray) -> np.ndarray:
    '''
        Функция взаимодействия агентов по правилу разделения

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]
                idx (int): индекс агента
                neigh_mask (np.ndarray): маска агентов, с которыми будет взаимодействовать изменяемый

        Возвращаемое значение:
                numpy.ndarray : массив из двух коэффицентов, на которые нужно изменить ускорения агентов
    '''

    x = np.median(boids[neigh_mask, 0] - boids[idx, 0]) #поиск среднего был заменен на медиану
    y = np.median(boids[neigh_mask, 1] - boids[idx, 1])


    return -np.array([x, y]) / ((x**2 + y**2)+1)

@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray) -> np.ndarray:
    '''
        Функция взаимодействия агентов по правилу сближения

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]
                idx (int): индекс агента
                neigh_mask (np.ndarray): маска агентов, с которыми будет взаимодействовать изменяемый

        Возвращаемое значение:
                a (numpy.ndarray): массив из двух коэффицентов, на которые нужно изменить ускорения агентов
    '''

    x = np.median(boids[neigh_mask, 2]) #поиск среднего был заменен на медиану
    y = np.median(boids[neigh_mask, 3])

    a = (np.array([x,y]) - boids[idx, 2:4])

    return a


@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             dzeros):
    '''
        Функция изменяет параметры агентов каждый фрейм относительно расстояния между ними

        Параметры:
                boids (numpy.ndarray): numpy массив размерности [n,6]
                perception (float): область видимости агентов
                coeffs (np.ndarray): коэффиценты поведения агентов
                asp (float): соотношение сторон окна
                dzeros (numpy.ndarray): массив размерноси [n,n]

    '''

    D = distance(boids, dzeros)
    N = boids.shape[0]

    mask = D < perception #взаимодействие объектов между собой было измененно на область видимости

    wal = walls(boids, asp)

    for i in prange(N):
        coh = cohesion(boids, i, mask[i])
        alg = alignment(boids, i, mask[i])
        sep = separation(boids, i, mask[i])

        x_rand = np.random.rand(1) #генерация шума
        y_rand = np.random.rand(1)
        if ((x_rand * 100) // 1) % 2 == 0:
            x_rand = -x_rand
        if ((y_rand * 100) // 1) % 2 == 0:
            y_rand = -y_rand


        a = coeffs[0] * coh[0] + coeffs[1] * alg[0] + coeffs[2] * sep[0] \
            + coeffs[3] * wal[i][0] + coeffs[4] * x_rand[0]
        b = coeffs[0] * coh[1] + coeffs[1] * alg[1] + coeffs[2] * sep[1] \
            + coeffs[3] * wal[i][1] + coeffs[4] * y_rand[0]

        boids[i, 4] = a
        boids[i, 5] = b


