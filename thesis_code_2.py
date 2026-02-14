import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

# =============================
# PARAMETERS
# =============================
M1 = 1.0
M2 = 0.5
K  = 50.0
C  = 1.0
l  = 1.0
g  = 9.81

# Optional forcing (set A=0 for no forcing)
A = 0.0
omega = 2.0

def x1(t):
    return A * np.sin(omega*t)

def dx1(t):
    return A * omega * np.cos(omega*t)

# =============================
# SYSTEM EQUATIONS
# =============================
def derivatives(t, y):
    x2, v2, x3, v3 = y

    Delta = x3 - x2
    denom = l**2 - Delta**2
    if abs(denom) < 1e-6:
        denom = 1e-6

    a2 = (
        -K*(x2 - x1(t))
        -C*(v2 - dx1(t))
        + M2*(g/l)*Delta
        - M2*(Delta/denom)*(v3 - v2)**2
    ) / M1

    a3 = (
        a2
        - (g/l)*Delta
        + (Delta/denom)*(v3 - v2)**2
    )

    return np.array([v2, a2, v3, a3])

# =============================
# RK4 STEP
# =============================
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# =============================
# SIMULATION SETTINGS
# =============================
t0 = 0.0
tf = 50.0
dt = 0.001

time = np.arange(t0, tf, dt)

# Initial conditions
y = np.array([0.0, 0.0, 0.2, 0.0])
solution = []

for t in time:
    solution.append(y)
    y = rk4_step(derivatives, t, y, dt)

solution = np.array(solution)

x2 = solution[:,0]
v2 = solution[:,1]
x3 = solution[:,2]
v3 = solution[:,3]

# =============================
# 1️⃣ TIME RESPONSE
# =============================
plt.figure()
plt.plot(time, x2)
plt.title("Block Position x2(t)")
plt.xlabel("Time")
plt.ylabel("x2")
plt.show()

# =============================
# 2️⃣ PHASE PORTRAIT
# =============================
plt.figure()
plt.plot(x2, v2)
plt.title("Phase Portrait (x2 vs v2)")
plt.xlabel("x2")
plt.ylabel("v2")
plt.show()

# =============================
# 3️⃣ FFT SPECTRUM
# =============================
X = fft(x2)
freq = fftfreq(len(time), dt)

plt.figure()
plt.plot(freq[:5000], np.abs(X[:5000]))
plt.title("Frequency Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()

# =============================
# 4️⃣ LYAPUNOV EXPONENT
# =============================
epsilon = 1e-8
y1 = np.array([0.0, 0.0, 0.2, 0.0])
y2 = y1 + np.array([epsilon, 0, 0, 0])

distances = []

for t in time:
    y1 = rk4_step(derivatives, t, y1, dt)
    y2 = rk4_step(derivatives, t, y2, dt)

    d = np.linalg.norm(y2 - y1)
    distances.append(d)

distances = np.array(distances)
lyap = np.mean(np.log(distances/epsilon)) / tf

print("Estimated Lyapunov Exponent:", lyap)

if lyap > 0:
    print("System is likely CHAOTIC")
elif abs(lyap) < 1e-3:
    print("System is likely QUASI-PERIODIC")
else:
    print("System is likely PERIODIC")

# =============================
# 5️⃣ POINCARÉ SECTION (if forced)
# =============================
if A != 0:
    T = 2*np.pi/omega
    poincare_x = []
    poincare_v = []

    for i, t in enumerate(time):
        if abs((t % T)) < dt:
            poincare_x.append(x2[i])
            poincare_v.append(v2[i])

    plt.figure()
    plt.scatter(poincare_x, poincare_v, s=5)
    plt.title("Poincaré Section")
    plt.xlabel("x2")
    plt.ylabel("v2")
    plt.show()

