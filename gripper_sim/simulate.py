"""
Parallel Gripper Grasping Simulation
=====================================
Simulates a parallel gripper picking an object off a table using MuJoCo.

Phases
------
1. OPEN    – fingers open, gripper above the object
2. DESCEND – gripper lowers to grasp height
3. CLOSE   – fingers close around the object
4. LIFT    – gripper rises, carrying the object
5. HOLD    – hold final position
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# ── paths ────────────────────────────────────────────────────────────────────
XML_PATH = os.path.join(os.path.dirname(__file__), "gripper.xml")

# ── actuator indices (must match <actuator> order in XML) ────────────────────
IDX_Z     = 0   # gripper_z     – vertical slide
IDX_LEFT  = 1   # left_close    – left finger
IDX_RIGHT = 2   # right_close   – right finger

# ── control targets for each phase ──────────────────────────────────────────
#  gripper_z range: -0.35 (low) … 0.1 (high)
#  finger range:     0.0 (open)  … 0.06 (closed)
PHASES = [
    # name,       z_target,   left,   right,  duration_s
    ("OPEN",      0.0,        0.0,    0.0,    0.5),
    ("DESCEND",  -0.27,       0.0,    0.0,    1.5),
    ("CLOSE",    -0.27,       0.05,   0.05,   1.0),
    ("LIFT",      0.0,        0.05,   0.05,   2.0),
    ("HOLD",      0.0,        0.05,   0.05,   1.5),
]


def set_ctrl(data: mujoco.MjData, z: float, left: float, right: float) -> None:
    data.ctrl[IDX_Z]     = z
    data.ctrl[IDX_LEFT]  = left
    data.ctrl[IDX_RIGHT] = right


def get_object_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "object_site")
    return data.site_xpos[site_id].copy()


def run_simulation(render: bool = True, record: bool = False):
    """
    Parameters
    ----------
    render : bool
        Open an interactive MuJoCo viewer window.
    record : bool
        Also record frames and save a video (requires mediapy).
    """
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    frames = []

    if record:
        import mediapy as media
        os.environ.setdefault("MUJOCO_GL", "egl")
        renderer = mujoco.Renderer(model, height=480, width=640)

    if render:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance  = 1.2
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0, 0, 0.55]

    print("\n=== Parallel Gripper Grasping Simulation ===\n")

    for phase_name, z_tgt, l_tgt, r_tgt, duration in PHASES:
        print(f"[{phase_name}]  z={z_tgt:+.3f}  fingers={l_tgt:.3f}")
        t_end = data.time + duration

        while data.time < t_end:
            set_ctrl(data, z_tgt, l_tgt, r_tgt)
            mujoco.mj_step(model, data)

            if record:
                renderer.update_scene(data, camera="track_cam" if "track_cam" in
                                      [model.cam(i).name for i in range(model.ncam)]
                                      else -1)
                frames.append(renderer.render())

            if render:
                viewer.sync()
                time.sleep(model.opt.timestep)   # real-time pacing

        obj_pos = get_object_pos(model, data)
        print(f"          object z = {obj_pos[2]:.4f} m")

    print("\n=== Simulation complete ===")

    if record and frames:
        out = os.path.join(os.path.dirname(__file__), "grasp.mp4")
        media.write_video(out, frames, fps=int(1 / model.opt.timestep / 10))
        print(f"Video saved to {out}")

    if render:
        print("Close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
        viewer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel gripper MuJoCo demo")
    parser.add_argument("--no-render", action="store_true",
                        help="Run headless (no viewer window)")
    parser.add_argument("--record",   action="store_true",
                        help="Record a video (grasp.mp4)")
    args = parser.parse_args()

    run_simulation(render=not args.no_render, record=args.record)
