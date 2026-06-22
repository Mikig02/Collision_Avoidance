import os
import numpy as np
import tensorflow as tf
from storm_env import StormEnv
import time
from load_storm_ddqn import load_keras3_weights_into

N_OBS = 50
N_ACTIONS = 11
TEST_DURATION_SEC = 300  # 5 minutes, as in the paper (Table 3)

WEIGHTS_PATH = os.path.expanduser('~/storm_dqn/storm_agents/storm_ddqn_best.weights.h5')
# WEIGHTS_PATH = os.path.expanduser('/home/miky/storm_ddqn_best.weights.beta.0.999.h5')


def build_qnet():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_OBS,)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(N_ACTIONS, activation='linear'),
    ])


def main():
    env = StormEnv()
    env.max_steps = 9999
    env._collision_dist = 0.25

    print(f"Loading weights from: {WEIGHTS_PATH}")

    model = build_qnet()

    # Load the policy weights. Choose ONE of the following:
    #   model.load_weights(WEIGHTS_PATH)                  # weights saved with Keras 2
    #   load_keras3_weights_into(model, WEIGHTS_PATH)     # weights saved with Keras 3

    obs, _ = env.reset()

    collisions      = 0
    total_steps     = 0
    episode_steps   = 0
    t_start         = time.time()

    print(f"Starting greedy test — duration: {TEST_DURATION_SEC}s")
    print("=" * 50)

    while True:
        elapsed = time.time() - t_start
        if elapsed >= TEST_DURATION_SEC:
            break

        # Pure greedy action — no exploration, as in the paper (§5.2)
        q_values = model(obs[np.newaxis], training=False).numpy()[0]
        action   = int(np.argmax(q_values))

        obs, reward, terminated, truncated, _ = env.step(action)
        total_steps  += 1
        episode_steps += 1

        if terminated:
            # `terminated` = collision (reward == -1000)
            collisions += 1
            print(f"  [t={elapsed:5.1f}s] Collision #{collisions} "
                  f"after {episode_steps} steps")
            obs, _ = env.reset()
            episode_steps = 0

        elif truncated:
            # Max steps reached without collision — not counted as a collision
            obs, _ = env.reset()
            episode_steps = 0

    elapsed_total = time.time() - t_start
    print("=" * 50)
    print(f"Test completed in {elapsed_total:.1f}s")
    print(f"Total collisions : {collisions}")
    print(f"Total steps      : {total_steps}")
    print(f"Avg steps/s      : {total_steps / elapsed_total:.1f}")
    env.close()


if __name__ == '__main__':
    main()
