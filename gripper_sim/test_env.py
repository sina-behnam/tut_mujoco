"""Quick sanity check: run a few random episodes with random actions."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from gripper_env import GripperEnv, OBJECT_TYPES

def run_episode(env, seed):
    obs, info = env.reset(seed=seed)
    print(f"\n  object={info['object_type']:10s}  obs shape={obs.shape}")

    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    status = "SUCCESS" if info["is_success"] else ("truncated" if not terminated else "fail")
    print(f"  steps={step+1:3d}  lift={info['lift_height']:+.4f} m  "
          f"total_r={total_reward:+7.2f}  [{status}]")


if __name__ == "__main__":
    print("=== GripperEnv sanity check ===")

    # 1) fixed object types
    for obj in OBJECT_TYPES:
        env = GripperEnv(object_type=obj, max_episode_steps=200)
        print(f"\n[{obj}]")
        for ep in range(2):
            run_episode(env, seed=ep)

    # 2) random object selection
    print("\n[random]")
    env = GripperEnv(object_type="random", max_episode_steps=200)
    for ep in range(4):
        run_episode(env, seed=ep)

    # 3) rendering
    env_r = GripperEnv(render_mode="rgb_array", max_episode_steps=5)
    env_r.reset()
    frame = env_r.render()
    print(f"\nRender frame shape: {frame.shape}  dtype={frame.dtype}")
    env_r.close()

    print("\n=== All checks passed ===")
