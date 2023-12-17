import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

plt.ion()

# Parametry symulacji
Lx = Ly = 1.0  # Rozmiary obszaru [m]
Nx = Ny = 100  # Liczba punktów siatki w każdym kierunku [1]
Nt = 100  # Liczba kroków czasowych na sekundę [1]
std_dev = 0.2  # Odchylenie standardowe rozkładu początkowego
noise_intensity = 0.1  # Intensywność szumów w stanie początkowym
alpha = 2e-3  # Współczynnik wyrównania temperatur [m^2/s]
source = (0.5, 0.5)  # Położenie źródła ciepła [(m, m)]
source_power = 10  # Moc źródła ciepła [W]
FPS = 10  # Liczba klatek na sekundę symulacji [1]
T = 5  # Czas symulacji [s]
frame_time_real = 50  # Rzeczywisty czas klatki [ms]

# Wyliczone parametry
frames = FPS * T  # Liczba klatek
frame_time_virt = 1 / FPS  # Czas klatki symulacji [s]

# Krok czasowy i przestrzenny
dt = frame_time_virt / Nt
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Tworzenie siatki
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Początkowy rozkład temperatury
u: np.ndarray = None


def reset_data():
    global u
    u = np.exp(-((X - source[0]) ** 2 + (Y - source[1]) ** 2) / (2 * std_dev**2))
    u[:, :] += noise_intensity * np.random.rand(Nx, Ny)  # Dodanie szumu


reset_data()

# Indeks źródła
source_loc = int(source[0] / Lx * Nx), int(source[1] / Ly * Ny)

# Przygotowanie wykresu
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

surf = None


def update(*_):
    for _ in np.arange(0, Nt):
        # Powielenie wartości brzegowych
        u[+0, :] = u[+1, :]
        u[-1, :] = u[-2, :]
        u[:, +0] = u[:, +1]
        u[:, -1] = u[:, -2]

        d2u_dx2 = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
        d2u_dy2 = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        u[1:-1, 1:-1] += alpha * dt * (d2u_dx2 + d2u_dy2)

        # Źródło ciepła
        u[source_loc] += source_power * dt

    global surf
    if surf is not None:
        surf.remove()
    surf = ax.plot_surface(X, Y, u, cmap="viridis", rstride=5, cstride=5, alpha=0.7)


update()

ax.set_xlabel("X $[m]$")
ax.set_ylabel("Y $[m]$")
ax.set_zlabel("Temperature $[K]$")
ax.set_box_aspect(aspect=None, zoom=1)
ax.set_zlim(0, 1)
fig.subplots_adjust(left=0.25, bottom=0.25)

# Slidery
# # Step slider
axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
time_slider = Slider(
    ax=axtime,
    label="Time",
    valmin=0.0,
    valmax=T,
    valstep=frame_time_virt,
    valinit=0.0,
    valfmt="%.2f $[s]$",
)
time_slider.set_active(False)
time_slider.on_changed(update)

# # Alpha slider
axalpha = fig.add_axes([0.18, 0.25, 0.0225, 0.63])
alpha_slider = Slider(
    ax=axalpha,
    label="Thermal\ndiffusivity",
    valmin=1e-3,
    valmax=9e-3,
    valinit=2e-3,
    orientation="vertical",
    valfmt="%.5f $[\\frac{m^2}{s}]$",
)


def update_alpha(val):
    global alpha
    alpha = val


alpha_slider.on_changed(update_alpha)

# # Moc źródła slider
axsource = fig.add_axes([0.08, 0.25, 0.0225, 0.63])
source_slider = Slider(
    ax=axsource,
    label="Source\nPower",
    valmin=0,
    valmax=20,
    valinit=10,
    orientation="vertical",
    valfmt="%.2f $[W]$",
)


def update_source(val):
    global source_power
    source_power = val


source_slider.on_changed(update_source)


# Guziki
# # Reset
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="0.975")


def reset(event):
    global ani
    if ani is not None and ani.event_source is not None:
        ani.event_source.stop()
    reset_data()
    time_slider.reset()
    source_slider.reset()
    alpha_slider.reset()


button.on_clicked(reset)

# # Animate
ani = None
animationax = fig.add_axes([0.65, 0.025, 0.1, 0.04])
animate_button = Button(animationax, "Animate", hovercolor="0.975")


def animateAction(event):
    reset_data()
    time_slider.reset()

    def animate(frame):
        time_slider.set_val(time_slider.val + frame_time_virt)

    global ani
    if ani is not None and ani.event_source is not None:
        ani.event_source.stop()
    ani = FuncAnimation(
        fig, animate, frames=range(frames - 1), interval=frame_time_real, repeat=False
    )


animate_button.on_clicked(animateAction)

plt.show(block=True)
