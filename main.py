import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

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
STEPS = 100 # Steps in animation
STEP_SPEED = 50 # Step speed in animation [ms]

frame_time = 1 / FPS

# Krok czasowy i przestrzenny
dt = None
dx = None
dy = None

# Tworzenie siatki
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Warunki początkowe
u: np.ndarray = None

def reset_data():
    global u, dt, dx, dy
    dt = frame_time / Nt
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    u = np.exp(
        -((X - source[0]) ** 2 + (Y - source[1]) ** 2) / (2 * std_dev**2)
    )
    u[:, :] += noise_intensity * np.random.rand(Nx, Ny)  # Dodanie szumu

reset_data()

# Indeks źródła
source_loc = int(source[0] / Lx * Nx), int(source[1] / Ly * Ny)

# Przygotuj wykres
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

surf = None

def update(value=0, initial=False):
    print(value)
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
    if not initial:
        global surf
        surf.remove()
        surf = ax.plot_surface(X, Y, u, cmap="viridis", rstride=5, cstride=5, alpha=0.7)

update(0, True)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Temperatura")
ax.set_box_aspect(aspect=None, zoom=1)
ax.set_zlim(0, 1)
surf = ax.plot_surface(X, Y, u, cmap="viridis", rstride=5, cstride=5, alpha=0.7)
fig.subplots_adjust(left=0.25, bottom=0.25)

# Slidery
## Step slider
axstep = fig.add_axes([0.25, 0.1, 0.65, 0.03])
step_slider = Slider(
    ax=axstep,
    label='Step',
    valmin=0,
    valmax=STEPS,
    valinit=0,
)
step_slider.set_active(False)
step_slider.on_changed(update)

## Alpha slider
axalpha = fig.add_axes([0.15, 0.25, 0.0225, 0.63])
alpha_slider = Slider(
    ax=axalpha,
    label="Alpha",
    valmin=1e-3,
    valmax=9e-3,
    valinit=2e-3,
    orientation="vertical"
)

def update_alpha(val):
    global alpha
    alpha = val

alpha_slider.on_changed(update_alpha)

## Moc źródła slider
axsource = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
source_slider = Slider(
    ax=axsource,
    label="Source Power",
    valmin=0,
    valmax=20,
    valinit=10,
    orientation="vertical"
)

def update_source(val):
    global source_power
    source_power = val

source_slider.on_changed(update_source)


# Guziki
## Reset
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    global ani
    if ani is not None and ani.event_source is not None:
        ani.event_source.stop()
    reset_data()
    step_slider.reset()
    source_slider.reset()
    alpha_slider.reset()
button.on_clicked(reset)

## Animate
ani = None
animationax = fig.add_axes([0.65, 0.025, 0.1, 0.04])
animate_button = Button(animationax, 'Animate', hovercolor='0.975')
def animateAction(event):
    reset_data()
    step_slider.reset()
    def animate(frame):
        step_slider.set_val(step_slider.val + 1)
    global ani
    if ani is not None and ani.event_source is not None:
        ani.event_source.stop()
    ani = FuncAnimation(fig, animate, frames=range(STEPS - 1), interval=STEP_SPEED, repeat=False)
animate_button.on_clicked(animateAction)

plt.show(block=True)