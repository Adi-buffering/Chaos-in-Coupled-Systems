import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS
# -------------------------
M1 = 1.0      # mass of block
M2 = 0.5      # mass of pendulum
K  = 50.0     # spring constant
C  = 1.0      # damping coefficient
l  = 1.0      # string length
g  = 9.81     # gravity

# Base motion (earthquake input)
def x1(t):
    return 0.0   # change if needed

def dx1(t):
    return 0.0   # derivative of base motion


# -------------------------
# SYSTEM OF FIRST ORDER ODEs
# -------------------------
def derivatives(t, y):
    # y = [x2, v2, x3, v3]
    x2, v2, x3, v3 = y

    Delta = x3 - x2

    # Prevent singularity
    denom = l**2 - Delta**2
    if abs(denom) < 1e-6:
        denom = 1e-6

    # Compute x2 acceleration
    a2 = (
        -K*(x2 - x1(t))
        -C*(v2 - dx1(t))
        + M2*(g/l)*Delta
        - M2*(Delta/denom)*(v3 - v2)**2
    ) / M1

    # Compute x3 acceleration
    a3 = (
        a2
        - (g/l)*Delta
        + (Delta/denom)*(v3 - v2)**2
    )

    return np.array([v2, a2, v3, a3])


# -------------------------
# RK4 IMPLEMENTATION
# -------------------------
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


# -------------------------
# SIMULATION SETTINGS
# -------------------------
t0 = 0.0
tf = 20.0
dt = 0.0001

time = np.arange(t0, tf, dt)

# Initial conditions
# x2, v2, x3, v3
y = np.array([0.0, 0.0, 0.2, 0.0])

solution = []

# Time integration
for t in time:
    solution.append(y)
    y = rk4_step(derivatives, t, y, dt)

solution = np.array(solution)

# -------------------------
# PLOT RESULTS
# -------------------------
plt.figure()
plt.plot(time, solution[:,0])
plt.title("Block Position x2(t)")
plt.xlabel("Time")
plt.ylabel("x2")
plt.show()

plt.figure()
plt.plot(time, solution[:,2])
plt.title("Pendulum Bob Position x3(t)")
plt.xlabel("Time")
plt.ylabel("x3")
plt.show()


plt.figure()
plt.plot(solution[:,0], solution[:,1])
plt.xlabel("x2")
plt.ylabel("v2")
plt.title("Phase Portrait (Block)")
plt.show()
