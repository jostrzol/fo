import numpy as np
import matplotlib.pyplot as plt
import time
import math

plt.ion()

# Parametry symulacji
Lx = Ly = 1.0  # Rozmiary obszaru
Nx = Ny = 100  # Liczba punktów siatki w każdym kierunku
Nt = 100  # Liczba kroków czasowych w jednej klatce
std_dev = 0.2  # Odchylenie standardowe rozkładu początkowego
noise_intensity = 0.1  # Intensywność szumów w stanie początkowym
alpha = 2e-3  # Współczynnik przewodzenia ciepła
source = (0.5, 0.5)  # Położenie źródła ciepła
source_power = 10  # Moc źródła ciepła
FPS = 8  # Frames per second

frame_time = 1 / FPS

# Krok czasowy i przestrzenny
dt = frame_time / Nt
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Tworzenie siatki
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Warunki początkowe
u: np.ndarray = np.exp(
    -((X - source[0]) ** 2 + (Y - source[1]) ** 2) / (2 * std_dev**2)
)
u[:, :] += noise_intensity * np.random.rand(Nx, Ny)  # Dodanie szumu

# Indeks źródła
source_loc = int(source[0] / Lx * Nx), int(source[1] / Ly * Ny)

# Przygotuj wykres
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Warunek końca
more_work = True


def on_close(_):
    global more_work
    more_work = False


fig.canvas.mpl_connect("close_event", on_close)

# Funcja kroku
next_tick = time.time()
while more_work:
    missed_frames = math.floor((time.time() - next_tick) / frame_time)
    if missed_frames != 0:
        print(f"missed {missed_frames} frames")
    next_tick = next_tick + (missed_frames + 1) * frame_time

    for _ in np.arange(0, Nt):
        # Powielenie wartości brzegowych
        u[+0, :] = u[+1, :]
        u[-1, :] = u[-2, :]
        u[:, +0] = u[:, +1]
        u[:, -1] = u[:, -2]

        du2_dx2 = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
        du2_dy2 = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        u[1:-1, 1:-1] += alpha * dt * (du2_dx2 + du2_dy2)

        # Źródło ciepła
        u[source_loc] += source_power * dt

    ax.clear()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Temperatura")
    ax.set_box_aspect(aspect=None, zoom=1)
    ax.set_zlim(0, 1)
    ax.plot_surface(X, Y, u, cmap="viridis", rstride=5, cstride=5, alpha=0.7)
    fig.canvas.flush_events()
    interval = next_tick - time.time()
    interval = interval if interval > 0 else 0
    time.sleep(interval)
