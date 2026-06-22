import time
import math
import random
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tf2_msgs.msg import TFMessage

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from ros_gz_interfaces.srv import SetEntityPose


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LIDAR        = 50
RANGE_MAX      = 5.0
N_ACTIONS      = 11
LINEAR_VEL     = 0.2
ANGULAR_VELS   = [-0.8 + 0.16 * i for i in range(11)]  # -0.8 .. +0.8
COLLISION_DIST = 0.25
REWARD_STEP      =    5
REWARD_COLLISION = -1000

# Spawn presets for the other maps (kept for reference):
#
# Training map:
#   (-4.38,  3.00,  1.50), ( 2.10, -0.05,  0.00), (-0.25,  3.31, -0.60),
#   (-0.82, -3.86,  0.00), (-2.69,  4.14,  0.00)
#
# test_map_3:
#   (-3.8,  3.3,   0.0), ( 4.8,  3.3,  -1.5708),
#   ( 4.8, -0.3,   3.1416), (-3.5, -3.0,   0.0)

SPAWN_POSITIONS = [            # Spawn for test_map_1
    (-2.3, -3.0, 1.5),
]

# ---------------------------------------------------------------------------
# Continuous random spawn — replicates the paper's "random position in the 3D
# world". Geometry is extracted from the .world file: sample uniformly inside
# the walls' bounding box and reject points inside/near walls or cylinders
# (analytic exclusion), then confirm with the LiDAR. Walls and cylinders are
# excluded by construction.
# ---------------------------------------------------------------------------

# _SEEDS = np.load('semi_corridoio.npy')
# SPAWN_PERTURB = 0.15      # spawn for training map
SPAWN_Z = 0.5


# ---------------------------------------------------------------------------
# Internal ROS 2 node
# ---------------------------------------------------------------------------
class _RosNode(Node):

    def __init__(self):
        super().__init__('storm_dqn')

        # Publisher: /cmd_vel
        self.pub_vel = self.create_publisher(
            Twist, '/model/storm/cmd_vel', 10)

        # Subscriber: /scan
        self._scan      = None
        self._scan_lock = threading.Lock()
        self.create_subscription(
            LaserScan, '/model/storm/scan', self._cb_scan, 10)

        # Service client: set_pose
        self.cli_pose = self.create_client(
            # SetEntityPose, '/world/training_map/set_pose')
            # SetEntityPose, '/world/test_map_3/set_pose')
            SetEntityPose, '/world/test_map_1/set_pose')

        self._pose = None
        self._pose_lock = threading.Lock()
        self.create_subscription(
            TFMessage, '/model/storm/tf', self._cb_tf, 10)

    def _cb_tf(self, msg):
        for t in msg.transforms:
            if t.child_frame_id == 'storm' or 'storm' in t.child_frame_id:
                with self._pose_lock:
                    self._pose = (t.transform.translation.x,
                                  t.transform.translation.y,
                                  t.transform.translation.z)

    def get_pose(self):
        with self._pose_lock:
            return self._pose

    def _cb_scan(self, msg):
        with self._scan_lock:
            self._scan = msg

    def get_scan(self):
        with self._scan_lock:
            return self._scan

    def pub_cmd(self, lin, ang):
        msg = Twist()
        msg.linear.x  = float(lin)
        msg.angular.z = float(ang)
        self.pub_vel.publish(msg)

    def set_pose(self, x, y, z, qx, qy, qz, qw):
        """Call set_pose with retry + backoff.

        This avoids the timeout/blocking issue previously seen in MATLAB with
        call(..., 'Timeout', 60).
        """
        req = SetEntityPose.Request()
        req.entity.name        = 'storm'
        req.entity.type        = 1
        req.pose.position.x    = float(x)
        req.pose.position.y    = float(y)
        req.pose.position.z    = float(z)
        req.pose.orientation.x = float(qx)
        req.pose.orientation.y = float(qy)
        req.pose.orientation.z = float(qz)
        req.pose.orientation.w = float(qw)

        for attempt in range(5):
            if not self.cli_pose.wait_for_service(timeout_sec=2.0):
                time.sleep(0.5 * (attempt + 1))
                continue
            future   = self.cli_pose.call_async(req)
            deadline = time.time() + 5.0
            while not future.done() and time.time() < deadline:
                time.sleep(0.05)
            if future.done():
                return   # success
            time.sleep(0.5 * (attempt + 1))

        # If we get here, all attempts failed — the robot stays where it is
        # (worst case: the episode starts from the previous position).


# ---------------------------------------------------------------------------
# StormEnv — gymnasium.Env
# Direct translation of StormEnv.m, method by method.
# ---------------------------------------------------------------------------
class StormEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # rlNumericSpec([50 1], LowerLimit 0, UpperLimit 5)
        self.observation_space = spaces.Box(
            low  = np.zeros(N_LIDAR, dtype=np.float32),
            high = np.full(N_LIDAR, RANGE_MAX, dtype=np.float32),
            dtype= np.float32)

        # rlFiniteSetSpec(1:11) -> Discrete(11), actions 0..10
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._collision_dist = COLLISION_DIST
        self.step_count = 0
        self.max_steps  = 1000   # MaxSteps from StormEnv.m

        # Start ROS 2 and run the spinner in a separate thread
        if not rclpy.ok():
            rclpy.init()
        self._node = _RosNode()
        threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True).start()

        # Wait for the first scan — equivalent to MATLAB's pause(5)
        print('StormEnv: waiting for the first LiDAR message...')
        t0 = time.time()
        while self._node.get_scan() is None:
            time.sleep(0.1)
            if time.time() - t0 > 10.0:
                raise RuntimeError(
                    'Timeout: no message on /model/storm/scan.\n'
                    'Make sure Gazebo is running via sim_launch.py')
        print('StormEnv: connected to ROS 2 Humble.')

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------
    def step(self, action):
        self._send_action(action)
        time.sleep(0.1)          # pause(0.1)

        obs = self._get_obs()

        if self._check_collision(obs):
            reward     = REWARD_COLLISION
            terminated = True
            self._stop()
        else:
            reward     = REWARD_STEP
            terminated = (self.step_count >= self.max_steps)

        self.step_count += 1
        return obs, reward, terminated, False, {}

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Stop fully BEFORE repositioning: send several zero commands with a
        # wait, so Gazebo bleeds off the leftover velocity from the previous
        # episode.
        for _ in range(3):
            self._stop()
            time.sleep(0.1)
        self._reset_pose()
        self.step_count = 0
        # After repositioning: stop again, let the physics settle, then
        # re-check that the spawn is still clear (not drifted into a wall).
        self._stop()
        time.sleep(0.5)
        self._stop()
        return self._get_obs(), {}

    def close(self):
        self._stop()
        self._node.destroy_node()

    # -----------------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------------

    def _get_obs(self):
        """getObservation()"""
        msg = self._node.get_scan()
        if msg is None:
            return np.full(N_LIDAR, RANGE_MAX, dtype=np.float32)
        raw = np.array(msg.ranges, dtype=np.float32)
        raw[np.isinf(raw) | np.isnan(raw)] = RANGE_MAX
        raw = np.clip(raw, 0.0, RANGE_MAX)
        idx = np.round(np.linspace(0, len(raw) - 1, N_LIDAR)).astype(int)
        return raw[idx]

    def _check_collision(self, obs):
        """checkCollision()"""
        return bool(np.any(obs < self._collision_dist))

    def _send_action(self, action_idx):
        """sendAction() — action_idx is 0-based (0..10)"""
        self._node.pub_cmd(LINEAR_VEL, ANGULAR_VELS[int(action_idx)])

    def _stop(self):
        """stopRobot()"""
        self._node.pub_cmd(0.0, 0.0)

    def _reset_pose(self):
        for _ in range(20):
            x, y, yaw = random.choice(SPAWN_POSITIONS)
            qz, qw = math.sin(yaw/2.0), math.cos(yaw/2.0)
            self._node.set_pose(x, y, SPAWN_Z, 0.0, 0.0, qz, qw)
            self._stop()
            time.sleep(0.4)
            self._stop()
            if float(self._get_obs().min()) >= 0.30:
                return
        # Fallback: keep the last pose anyway.

    # -----------------------------------------------------------------------
    # Alternative spawn strategies (kept for reference, currently unused)
    # -----------------------------------------------------------------------

    # --- A) Perturbed spawn around precomputed seeds (training map) ---
    # def _reset_pose(self):
    #     for _ in range(50):
    #         sx, sy = _SEEDS[random.randint(0, len(_SEEDS) - 1)]
    #         x = sx + random.uniform(-SPAWN_PERTURB, SPAWN_PERTURB)
    #         y = sy + random.uniform(-SPAWN_PERTURB, SPAWN_PERTURB)
    #         yaw = random.uniform(-math.pi, math.pi)
    #         qz, qw = math.sin(yaw/2), math.cos(yaw/2)
    #         self._node.set_pose(x, y, 0.5, 0.0, 0.0, qz, qw)
    #         self._stop()
    #         time.sleep(0.4)
    #         self._stop()
    #         obs = self._get_obs()
    #         if float(obs.min()) >= 0.30:
    #             return
    #     sx, sy = _SEEDS[random.randint(0, len(_SEEDS) - 1)]
    #     self._node.set_pose(float(sx), float(sy), 0.5, 0.0, 0.0, 0.0, 1.0)

    # --- B) Spawn with best orientation (Feng et al., modified) ---
    # def _reset_pose(self):
    #     """resetRobotPose() with smart spawn (Feng et al., modified)."""
    #     for _ in range(50):
    #         # Pick a base seed and add perturbation
    #         sx, sy = _SEEDS[random.randint(0, len(_SEEDS) - 1)]
    #         x = sx + random.uniform(-SPAWN_PERTURB, SPAWN_PERTURB)
    #         y = sy + random.uniform(-SPAWN_PERTURB, SPAWN_PERTURB)
    #
    #         # 1. Temporarily place the robot with yaw=0 to read the surroundings
    #         self._node.set_pose(x, y, SPAWN_Z, 0.0, 0.0, 0.0, 1.0)
    #         self._stop()
    #         time.sleep(0.4)
    #         self._stop()
    #
    #         # 2. Read the LiDAR
    #         obs = self._get_obs()
    #
    #         # 3. FIRST FIX: raise the spawn clearance threshold from 0.30 to 0.45
    #         if float(obs.min()) >= 0.45:
    #             # 4. SECOND FIX: compute the optimal yaw toward free space
    #             best_idx = int(np.argmax(obs))  # index of the longest LiDAR ray
    #
    #             # The paper specifies a LiDAR with a 270-degree span
    #             angle_min = -135.0 * (math.pi / 180.0)  # -2.356 rad
    #             angle_max = 135.0 * (math.pi / 180.0)   # +2.356 rad
    #             angle_increment = (angle_max - angle_min) / len(obs)
    #
    #             # Angle (rad) corresponding to the best ray
    #             best_yaw = angle_min + (best_idx * angle_increment)
    #
    #             # Add a little random noise (~+/-15 deg = ~0.26 rad) for variety
    #             final_yaw = best_yaw + random.uniform(-0.26, 0.26)
    #             qz = math.sin(final_yaw / 2.0)
    #             qw = math.cos(final_yaw / 2.0)
    #
    #             # 5. Re-apply the pose at the same (x, y) with the new orientation
    #             self._node.set_pose(x, y, SPAWN_Z, 0.0, 0.0, qz, qw)
    #             self._stop()
    #             time.sleep(0.1)  # short pause to let physics settle
    #             return           # spawn done, exit the loop
    #
    #     # Emergency fallback: if all 50 attempts fail, drop it on the base seed
    #     sx, sy = _SEEDS[random.randint(0, len(_SEEDS) - 1)]
    #     self._node.set_pose(float(sx), float(sy), SPAWN_Z, 0.0, 0.0, 0.0, 1.0)
