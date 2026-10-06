"""
Pathlines in a rotating flow
============================

Compute pathlines in a synthetic velocity field that rotates around the origin,
and plot them on top of the speed.
"""

import matplotlib.pyplot as plt
import numpy as np

from glacier_flow_tools.interpolation import velocity
from glacier_flow_tools.pathlines import compute_pathline

# %%
# Velocity field
# --------------
#
# A rigid rotation around the origin, in m/yr on a 20 km by 20 km grid.
# One revolution takes ``2 * pi / omega`` years.

x = np.linspace(-10_000.0, 10_000.0, 201)
y = np.linspace(-10_000.0, 10_000.0, 201)
X, Y = np.meshgrid(x, y)

omega = 0.01
Vx = -omega * Y
Vy = omega * X
speed = np.hypot(Vx, Vy)

# %%
# Pathlines
# ---------
#
# :func:`glacier_flow_tools.pathlines.compute_pathline` integrates the velocity
# field with an adaptive Runge-Kutta-Fehlberg method.
# :func:`glacier_flow_tools.interpolation.velocity` supplies the velocity at a
# point by bilinear interpolation. Here each pathline covers three quarters of
# a revolution.

starting_points = [[2_000.0, 0.0], [4_000.0, 0.0], [6_000.0, 0.0], [8_000.0, 0.0]]
end_time = 0.75 * 2 * np.pi / omega

pathlines = [
    compute_pathline(
        point,
        velocity,
        f_args=(Vx, Vy, x, y),
        start_time=0.0,
        end_time=end_time,
        hmin=0.1,
        hmax=5.0,
        tol=1e-3,
    )
    for point in starting_points
]

# %%
# Plot
# ----
#
# The first element of each result holds the points along the pathline.

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.pcolormesh(x / 1e3, y / 1e3, speed, cmap="viridis", shading="auto")
for point, pathline in zip(starting_points, pathlines):
    pts = pathline[0]
    ax.plot(pts[:, 0] / 1e3, pts[:, 1] / 1e3, color="white", lw=1.5)
    ax.plot(point[0] / 1e3, point[1] / 1e3, "o", color="white", ms=4)
ax.set_aspect("equal")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
fig.colorbar(im, ax=ax, label="Speed (m/yr)")
plt.show()

# %%
# Check
# -----
#
# In a rigid rotation every particle stays on its circle, so the distance from
# the origin should not change along a pathline.

for point, pathline in zip(starting_points, pathlines):
    radius = np.hypot(pathline[0][:, 0], pathline[0][:, 1])
    print(f"start radius {point[0]:6.0f} m, largest deviation {np.abs(radius - point[0]).max():.1e} m")
