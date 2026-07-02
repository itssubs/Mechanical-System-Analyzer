import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar, minimize


def get_validatedinput(prompt, cast_type, condition, error_msg):
    while True:
        try:
            value = cast_type(input(prompt))
            if condition(value):
                return value
                break
            else:
                print(error_msg)
        except:
            print(f"Please enter a valid {cast_type. __name__}")

#User inputs
mass = get_validatedinput("Enter the mass of the spring mass damper system: ", float, lambda x : x > 0, "Mass should be greater than 0")
stiffness = get_validatedinput("Enter the stiffness of the spring in N/m: ", float, lambda x : x > 0, "Stiffness should be greater than 0")
damping = get_validatedinput("Enter the value of damping coefficient in Ns/m: ",float, lambda x : x >= 0, "Damping coefficient cannot be negative")
highest_operating_speed = get_validatedinput("Enter the highest speed the system needs to operate in RPM: ", float, lambda x : x > 0, "Operating speed should be greater than 0")
step = get_validatedinput("Enter the step size (speed increments): ", int, lambda x : x > 0, "Speed increments should be greater than 0")
unbalanced_mass = get_validatedinput("Enter the unbalanced mass of the spring mass damper system: ", float, lambda x : x > 0, "Mass should be greater than 0")
e = get_validatedinput("Enter the eccentricity of the spring mass damper system: ", float, lambda x : x > 0, "Eccentricity should be greater than 0")

#defining the speed mass damper system and the optimizing function
def spring_mass_damper_system(t, y, mass, stiffness, damping, F0, omega):
    x, v = y
    dxdt = v
    F = F0 * np.sin(omega * t)
    dvdt = (F - (stiffness * x) - (damping * v)) / mass
    return [dxdt, dvdt]

#optimizing damping coefficient
def amplitude(omega, c):
    F0 = unbalanced_mass * e * omega ** 2

    return F0 / (np.sqrt(((stiffness) - mass * omega ** 2) ** 2 + (c * omega) ** 2))

def objective(c):
    amps = []
    for omega in omega_values:
        amps. append(amplitude(omega, c[0]))
    return max(amps)


#solution of the initial value problem and amplitude calculation
speeds = np.arange(0, highest_operating_speed + step, step)
amplitudes = []
time = np.linspace(0, 20, 4000)
count = 1
omega_values = []
for speed in speeds:
    omega = speed * 2 * np.pi / 60
    omega_values.append(omega)
    F0 = unbalanced_mass * e * omega ** 2
    solution = solve_ivp(spring_mass_damper_system, [0, 20], [0, 0], t_eval = time, args = (mass, stiffness, damping, F0, omega))

    displacement = solution.y[0]
    steady = displacement[-1000:]
    amplitude = (np.max(steady) - np.min(steady)) / 2
    amplitudes.append(amplitude)

#conversion to numpy array and finding the position of the maximum 
amplitudes = np.array(amplitudes)
speeds = np.array(speeds)
position_of_max_amplitude = np.argmax(amplitudes)

#natural frequency and model interpolation to find the value of the theoritecal resonance point amplitude checking for the flagged case where the resonant frequency is +- 10 %
model = interp1d(speeds, amplitudes, kind = "cubic", fill_value = 'extrapolate')

#function for finding 
def natural_fre(omega):
    return stiffness - mass * omega ** 2

natural = root_scalar(natural_fre, bracket = [0, 2 * np.sqrt(stiffness/ mass)])
natural_freq = natural.root
natural_rpm = 60 * natural_freq / (2 * np.pi)

optimized_c = minimize(objective, x0 = [damping], bounds = [(0, 1000)])
amplitudes_optimized = []
for speed in speeds:
    omega = speed * 2 * np.pi / 60
    F0 = unbalanced_mass * e * omega ** 2
    solution_optimized = solve_ivp(spring_mass_damper_system, [0, 20], [0, 0], t_eval = time, args = (mass, stiffness, optimized_c.x[0], F0, omega))
    displacement_optimized = solution_optimized.y[0]
    steady_optimized = displacement_optimized[-1000:]
    amplitude_optimized = (np.max(steady_optimized) - np.min(steady_optimized)) / 2
    amplitudes_optimized.append(amplitude_optimized)

#critical speeds and their simulation

critical_speed = speeds[np.argmax(amplitudes)]
critical_omega = critical_speed * 2 * np.pi / 60
F0_critical = unbalanced_mass * e * critical_omega ** 2
solution_critical = solve_ivp(
    spring_mass_damper_system,
    [0,20],
    [0,0],
    t_eval=time,
    args=(mass, stiffness, damping, F0_critical, critical_omega)
)

#status 
dangerous_speeds = speeds[
    (speeds >= 0.9 * natural_rpm) &
    (speeds <= 1.1 * natural_rpm)
]
if len(dangerous_speeds) > 0:
    status = "DANGEROUS"
else:
    status = "SAFE"
#printing the results
print('=' * 60)
print('VIBRATION ANALYSIS REPORT')
print('=' * 60)

print("\n\nMachine Properties")
print(f"Mass = {mass} kg")
print(f"Spring Constant = {stiffness} N/m")
print(f"Damping Constant = {damping} Ns/m")

print('-' * 60)

print(f"\nNatural Frequency \n{natural_freq:.3f} rad/s \n{natural_rpm:.3f} RPM")

print('-' * 60)
print(f"\nTheoritical Peak \nOccurs at \n{natural_rpm:.3f} RPM \nAmplitude \n{model(natural_rpm):.3f} m")
print(f"\nActual Peak \nOccurs at \n{speeds[position_of_max_amplitude]} RPM \nAmplitude \n{amplitudes.max()} m")

print('-' * 60)
print(f"\nDanger Zone \n{0.9 * natural_rpm:.3f} RPM \nTo \n {1.1 * natural_rpm:.3f} RPM \nOperating Range \n0 - {highest_operating_speed:.3f} RPM \nSTATUS \n{status}")
print('-' * 60)

print(f"\nOptimization \nOriginal damping \n{damping} Ns/m \nOptimized damping \n{optimized_c.x[0]} Ns/m \nMaximum amplitude \nBefore \n{amplitudes.max():.3f} m \nAfter \n{max(amplitudes_optimized):.3f} m \nReduction \n{(abs(amplitudes.max() - max(amplitudes_optimized)) / amplitudes.max()) * 100:.3f} '%'")
print('-' * 60)
print(f"\nRecommendation")
if status.lower() == 'safe':
    print('Current operating Range is acceptable.')
else:
    print(f"Change the damping to \n{optimized_c.x[0]:.3f} Ns/m \nAvoid operating continuously between \n{0.9 * natural_rpm:.3f} To {1.1 * natural_rpm:.3f} RPM")
print('=' * 60)
print("Analysis completed successfully.")
print('=' * 60)

# graph plotting speed vs amplitude
fig, axs = plt.subplots(2,1, figsize= (6,6))
axs[0].plot(solution_critical.t, solution_critical.y[0], color = 'red', label = "Displacement plot at critical speed")
axs[0].set_ylabel("Displacement (m)")
axs[0].set_xlabel("Time (s)")
axs[0].set_title("Displacement vs Time Plot")
axs[0].legend()
axs[0].grid(True)
axs[1].plot(solution_critical.t, solution_critical.y[1], color = 'blue', label = "Velocity plot at critical speed")
axs[1].set_ylabel("Velocity (m/s)")
axs[1].set_xlabel("Time (s)")
axs[1].set_title("Velocity vs Time Plot")
axs[1].legend()
axs[1].grid(True)
plt.show()

plt.figure(figsize = (6,6))
plt.plot(speeds, amplitudes, color = "blue", label = f"Amplitude at given value of 'c = {damping}'")
plt.plot(speeds, amplitudes_optimized, color = "green", label = f"Amplitude at optimized value of 'c = {optimized_c.x[0]:.2f}'")
plt.axvline(natural_rpm, color = 'red', linestyle = "--", label = "Natural frequency of the system")
plt.plot(speeds[position_of_max_amplitude], amplitudes.max(), marker = 'x', color = 'red', label = "Actual Resonance point with maximum amplitude", linewidth = 2)
plt.annotate(text = f'({speeds[position_of_max_amplitude]:.3f}, {amplitudes.max():.3f})', xy = (speeds[position_of_max_amplitude], amplitudes.max()), xytext = (speeds[position_of_max_amplitude] + 10, amplitudes.max()), fontsize = 10, color = 'red')
plt.axvspan(0.9 * natural_rpm, 1.1 * natural_rpm, alpha = 0.2, color = 'red', label = "Dangerous zone")
plt.xlabel("Speed (RPM)")
plt.ylabel("Amplitude (m)")
plt.title("Speed v Amplitude plot")
plt.legend()
plt.grid(True)
plt.show()
