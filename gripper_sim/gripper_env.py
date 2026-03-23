"""
GripperEnv — MuJoCo parallel-gripper grasping environment
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

_XML = os.path.join(os.path.dirname(__file__), "gripper_multi.xml")

OBJECT_TYPES   = ["box", "cylinder", "sphere"]
# Rest z of object centre on the table (table surface = 0.42 m)
_OBJ_REST_Z    = {"box": 0.45, "cylinder": 0.455, "sphere": 0.448}
# Half-size in z used to compute "object bottom" height
_OBJ_HALF_Z    = {"box": 0.030, "cylinder": 0.035, "sphere": 0.028}

TABLE_Z        = 0.42     # top surface of the table (m)
LIFT_THRESHOLD = 0.08     # object must rise this far above rest to count as lifted

# actuator control ranges (must match XML)
_CTRL_Z_RANGE  = np.array([-0.35, 0.10])
_CTRL_FG_RANGE = np.array([ 0.00, 0.06])

# reward weights
_W_REACH   = 1.0
_W_LIFT    = 10.0
_W_GRASP   = 2.0
_SUCCESS   = 50.0


def _scale(x: float, lo: float, hi: float) -> float:
    """Map x ∈ [-1, 1] → [lo, hi]."""
    return lo + (x + 1.0) * 0.5 * (hi - lo)


def _normalise(x: float, lo: float, hi: float) -> float:
    """Map x ∈ [lo, hi] → [0, 1]."""
    return (x - lo) / (hi - lo)


class GripperEnv(gym.Env):
    """
    Parallel-gripper grasping environment
    Parameters
    ----------
    object_type : str
        "box", "cylinder", "sphere", or "random" (default).
    render_mode : str or None
        "rgb_array" to enable pixel rendering (required by gymnasium).
    max_episode_steps : int
        Maximum number of steps before truncation.
    sim_steps_per_action : int
        How many MuJoCo timesteps to simulate per `step()` call.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        object_type: str = "random",
        render_mode: str | None = None,
        max_episode_steps: int = 500,
        sim_steps_per_action: int = 20,
    ):
        assert object_type in OBJECT_TYPES + ["random"], \
            f"object_type must be one of {OBJECT_TYPES + ['random']}"

        super().__init__()   

        self.object_type = object_type
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.sim_steps_per_action = sim_steps_per_action
        
        self._rng = np.random.default_rng()

        self.model = mujoco.MjModel.from_xml_path(_XML)
        self.data  = mujoco.MjData(self.model)

        self._renderer: mujoco.Renderer | None = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        # cache id 
        def _jid(n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,   n)
        def _sid(n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,    n)
        def _gid(n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,    n)

        # freejoint qpos addresses  (pos[0:3] + quat[3:7])
        self._qpos_adr = {
            obj: self.model.jnt_qposadr[_jid(f"free_{obj[:3]}")]
            for obj in OBJECT_TYPES
        }
        # gripper / finger joint qpos addresses
        self._qpos_z  = self.model.jnt_qposadr[_jid("gripper_slide_z")]
        self._qpos_lf = self.model.jnt_qposadr[_jid("left_finger_joint")]
        self._qpos_rf = self.model.jnt_qposadr[_jid("right_finger_joint")]

        # site ids
        self._site_obj = {obj: _sid(f"site_{obj[:3]}") for obj in OBJECT_TYPES}
        self._site_gc  = _sid("gripper_center")
        self._site_lt  = _sid("site_left_tip")
        self._site_rt  = _sid("site_right_tip")

        # geom ids (for contact checking)
        self._geom_obj  = {obj: _gid(f"geom_{obj[:3]}") for obj in OBJECT_TYPES}
        self._geom_ltip = _gid("left_tip")
        self._geom_rtip = _gid("right_tip")

        
        self.observation_space = spaces.Box(
            low  = np.array([0, 0, 0, -0.5, -0.5, 0.0, -2, -2, -2, 0, 0, 0], dtype=np.float32),
            high = np.array([1, 1, 1,  0.5,  0.5, 1.5,  2,  2,  2, 1, 1, 1], dtype=np.float32),
            dtype = np.float32,
        )
        self.action_space = spaces.Box(
            low  = np.full(3, -1.0, dtype=np.float32),
            high = np.full(3,  1.0, dtype=np.float32),
            dtype = np.float32,
        )

        self._active_obj: str = "box"
        self._step_count: int = 0

    

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment; returns (observation, info)."""
        # gymnasium contract: call super().reset(seed=seed) to seed self.np_random
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # pick which object to use this episode
        if self.object_type == "random":
            self._active_obj = self._rng.choice(OBJECT_TYPES)
        else:
            self._active_obj = self.object_type

        mujoco.mj_resetData(self.model, self.data)

        # place active object on table with a small random XY offset
        xy_noise = self._rng.uniform(-0.06, 0.06, size=2)
        self._place_object(self._active_obj, xy_noise)

        # park inactive objects far below the world
        for obj in OBJECT_TYPES:
            if obj != self._active_obj:
                self._park_object(obj)

        # settle physics for a moment with no control
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        """
        Apply action, step simulation.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[0] = _scale(float(action[0]), *_CTRL_Z_RANGE)
        self.data.ctrl[1] = _scale(float(action[1]), *_CTRL_FG_RANGE)
        self.data.ctrl[2] = _scale(float(action[2]), *_CTRL_FG_RANGE)

        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)

        obs        = self._get_obs()
        reward     = self._compute_reward()
        terminated = self._is_success() or self._is_failure()
        truncated  = self._step_count >= self.max_episode_steps
        info       = self._get_info()

        self._step_count += 1
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    

    def _place_object(self, obj: str, xy_noise: np.ndarray):
        adr = self._qpos_adr[obj]
        self.data.qpos[adr + 0] = float(xy_noise[0])   # x
        self.data.qpos[adr + 1] = float(xy_noise[1])   # y
        self.data.qpos[adr + 2] = _OBJ_REST_Z[obj]     # z
        self.data.qpos[adr + 3] = 1.0                  # quat w
        self.data.qpos[adr + 4:adr + 7] = 0.0          # quat xyz
        mujoco.mj_forward(self.model, self.data)

    def _park_object(self, obj: str):
        adr = self._qpos_adr[obj]
        self.data.qpos[adr + 0] = 0.0
        self.data.qpos[adr + 1] = 0.0
        self.data.qpos[adr + 2] = -5.0                 # far below world
        self.data.qpos[adr + 3] = 1.0
        self.data.qpos[adr + 4:adr + 7] = 0.0

    def _obj_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._site_obj[self._active_obj]].copy()

    def _obj_vel(self) -> np.ndarray:
        """Linear velocity of the active object body."""
        adr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                              f"free_{self._active_obj[:3]}")
        ]
        return self.data.qvel[adr:adr + 3].copy()

    def _gripper_center(self) -> np.ndarray:
        return self.data.site_xpos[self._site_gc].copy()

    def _grasp_bonus(self) -> float:
        """Returns 1.0 if both finger tips are in contact with the active object."""
        obj_geom   = self._geom_obj[self._active_obj]
        l_contact  = False
        r_contact  = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if obj_geom in (g1, g2):
                if self._geom_ltip in (g1, g2):
                    l_contact = True
                if self._geom_rtip in (g1, g2):
                    r_contact = True
        return 1.0 if (l_contact and r_contact) else 0.0

    def _get_obs(self) -> np.ndarray:
        obj_pos = self._obj_pos()
        obj_vel = self._obj_vel()

        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[OBJECT_TYPES.index(self._active_obj)] = 1.0

        obs = np.array([
            _normalise(float(self.data.qpos[self._qpos_z]),  *_CTRL_Z_RANGE),
            _normalise(float(self.data.qpos[self._qpos_lf]), *_CTRL_FG_RANGE),
            _normalise(float(self.data.qpos[self._qpos_rf]), *_CTRL_FG_RANGE),
            obj_pos[0], obj_pos[1], obj_pos[2],
            obj_vel[0], obj_vel[1], obj_vel[2],
            *one_hot,
        ], dtype=np.float32)
        return obs

    def _compute_reward(self) -> float:
        obj_pos   = self._obj_pos()
        gc_pos    = self._gripper_center()
        dist      = float(np.linalg.norm(gc_pos - obj_pos))
        lift      = max(0.0, obj_pos[2] - (_OBJ_REST_Z[self._active_obj] + 0.005))
        grasp     = self._grasp_bonus()

        reward = (
            - _W_REACH * dist
            + _W_LIFT  * lift
            + _W_GRASP * grasp
        )
        if self._is_success():
            reward += _SUCCESS
        return float(reward)

    def _is_success(self) -> bool:
        obj_z = self._obj_pos()[2]
        return obj_z > (_OBJ_REST_Z[self._active_obj] + LIFT_THRESHOLD)

    def _is_failure(self) -> bool:
        obj_z = self._obj_pos()[2]
        return obj_z < (TABLE_Z - 0.05)

    def _get_info(self) -> dict:
        obj_pos = self._obj_pos()
        rest_z  = _OBJ_REST_Z[self._active_obj]
        return {
            "object_type":   self._active_obj,
            "object_pos":    obj_pos,
            "lift_height":   float(obj_pos[2] - rest_z),
            "is_success":    self._is_success(),
            "step":          self._step_count,
        }

    def __repr__(self):
        return f"GripperEnv(object_type={self.object_type!r})"
