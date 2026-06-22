# random_spawn_generator.py — expands the valid anchor spawns into many more,
# using the geometry parsed from the .world file.
import math
import numpy as np

# Known-good spawn anchors (world coordinates)
ANCORE = [
    (-4.38,  3.00),
    ( 2.10, -0.05),
    (-0.25,  3.31),
    (-0.82, -3.86),
    (-2.69,  4.14),
    (-1.90, -0.15),
    (2.95, 4.46),
    (5.68, 4.17),
    (5.02, 1.64),
    (0.95, -3.02),
    (3.56, 0.66),
    (-2.54,-2.55),
    (1.02, -1.36),
    (-2.84, 1.34),

]

# Geometry from the .world (the mappa_2 model offset is already added in)
_OFF = (0.654951, 0.036824)
_CYL = [(-4.05+_OFF[0], -0.02+_OFF[1], 1.0),
        (-1.50+_OFF[0], -2.22+_OFF[1], 1.1)]
_WALLS = [  # (cx, cy, yaw, L, T) in local frame -> offset added below
    (-3.65705,-1.32130,-2.09440,1.25,0.15),(-3.97128,-2.45953,-1.62999,1.47617,0.15),
    (-4.86060,1.29248,2.87979,2.0,0.15),(-5.74101,3.06881,1.55662,3.22416,0.15),
    (-3.92788,4.60574,0.0,3.73266,0.15),(5.75719,3.04244,-1.5708,4.25,0.15),
    (4.61512,1.00004,3.13502,2.46071,0.15),(0.45284,-3.85548,1.02847,1.79371,0.15),
    (0.93572,-0.62107,0.0,0.15,0.15),(0.90637,-1.91274,-1.59399,2.68116,0.15),
    (2.22362,-0.81026,2.99574,2.75343,0.15),(3.49437,0.00409,1.59656,2.15775,0.15),
    (-2.26066,-1.31701,1.0472,1.25,0.15),(-1.97687,-0.14373,1.55819,1.54405,0.15),
    (-2.46311,1.2405,-0.96103,1.84812,0.15),(-3.67002,2.1299,-0.26212,1.64234,0.15),
    (-4.3907,2.87325,1.5708,1.25,0.15),(-3.35389,3.39689,-0.02543,2.2243,0.15),
    (-1.16927,2.71173,-0.53352,2.77532,0.15),(-0.28634,-0.83901,-1.5708,3.75,0.15),
    (-3.66429,-3.75833,-1.06706,1.58449,0.15),(0.91333,0.74127,2.96045,2.58925,0.15),
    (1.06335,2.95272,-2.44916,2.9688,0.15),(2.113,1.44655,1.5708,2.0,0.15),
    (3.23087,2.36716,-0.00393,2.38576,0.15),(3.26603,3.84811,-0.00393,2.38576,0.15),
    (4.3839,3.10324,-1.59453,1.63137,0.15),(-2.34465,-4.76466,-0.36755,2.25485,0.15),
    (0.78988,4.30405,-2.35619,2.5,0.15),(-1.08876,4.03947,2.64612,2.53204,0.15),
    (3.68896,5.11367,3.13133,4.28668,0.15),(-0.66693,-4.85547,0.39623,1.63911,0.15),
]
WALLS = [(cx+_OFF[0], cy+_OFF[1], yaw, L, T) for (cx,cy,yaw,L,T) in _WALLS]

CLEAR_WALL = 0.30    # minimum clearance from a wall
CLEAR_CYL  = 0.45    # minimum clearance from a cylinder


def dist_wall(px, py, wx, wy, yaw, L, T):
    """Distance from point (px, py) to a rectangular wall segment."""
    dx, dy = px-wx, py-wy
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx, ly = c*dx - s*dy, s*dx + c*dy
    ex, ey = max(abs(lx)-L/2,0.0), max(abs(ly)-T/2,0.0)
    return math.hypot(ex, ey)


def is_free(x, y):
    """True if (x, y) keeps the required clearance from all walls and cylinders."""
    for (cx,cy,r) in _CYL:
        if math.hypot(x-cx,y-cy) < r+CLEAR_CYL:
            return False
    for w in WALLS:
        if dist_wall(x,y,*w) < CLEAR_WALL:
            return False
    return True


# Generate candidates: a disc of points around every anchor, keeping only the
# ones that pass the clearance check.
semi = []

# Discs around each anchor
for (ax, ay) in ANCORE:
    for r in np.arange(0.0, 0.9, 0.2):
        for th in np.arange(0, 2*math.pi, math.pi/6):
            x, y = ax + r*math.cos(th), ay + r*math.sin(th)
            if is_free(x, y):
                semi.append((round(x,2), round(y,2)))

# Deduplicate
semi = list(set(semi))
arr = np.array(semi)
np.save('semi_corridoio.npy', arr)
print(f"Generated {len(arr)} valid spawns (from {len(ANCORE)} anchors)")

# Verification plot (if matplotlib is available)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    # Draw the walls
    for (wx,wy,yaw,L,T) in WALLS:
        c,s = math.cos(yaw), math.sin(yaw)
        plt.plot([wx-c*L/2, wx+c*L/2],[wy-s*L/2, wy+s*L/2],'b-',lw=2)
    for (cx,cy,r) in _CYL:
        th=np.linspace(0,2*math.pi,40); plt.plot(cx+r*np.cos(th),cy+r*np.sin(th),'gray')
    plt.scatter(arr[:,0],arr[:,1],s=10,c='red')
    plt.scatter([a[0] for a in ANCORE],[a[1] for a in ANCORE],s=80,c='green',marker='*')
    plt.gca().set_aspect('equal'); plt.title(f'{len(arr)} spawns')
    plt.savefig('semi_plot.png',dpi=100); print("plot saved to semi_plot.png")
except ImportError:
    pass
