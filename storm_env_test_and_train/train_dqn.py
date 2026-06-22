"""
train_dqn.py
DDQN for collision avoidance — same logic as Feng et al. (2021).

How to run:
    1. Launch Gazebo:   ros2 launch storm_description sim_launch.py
    2. In another terminal:
       source ~/storm_env/bin/activate
       cd ~/storm_dqn
       python3 train_dqn.py

Monitor with TensorBoard (third terminal):
    source ~/storm_env/bin/activate
    tensorboard --logdir ~/storm_dqn/logs
    -> open http://localhost:6006
"""

from datetime import datetime
import os
import random
import time
import collections
import numpy as np
import tensorflow as tf

from storm_env import StormEnv

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MAX_EPISODES          = 3000       # paper: 3000
MAX_STEPS_PER_EPISODE = 1000       # max steps per episode

BUFFER_SIZE           = 50000      # experience replay buffer
BATCH_SIZE            = 64
GAMMA                 = 0.99       # discount factor

LR                    = 0.0005     # Adam learning rate

TARGET_UPDATE_FREQ    = 200        # update the target network every N steps

# Epsilon-greedy — paper Eq. (3): e_{k+1} = beta * e_k
BETA                  = 0.997      # per-episode decay rate (0.999 = best in the paper)
EPSILON_START         = 1.0
EPSILON_MIN           = 0.05       # epsilon floors here

# Reward — paper Eq. (4)
REWARD_STEP      =     5           # no collision
REWARD_COLLISION = -1000           # collision

WINDOW_SIZE           = 50         # moving-average reward window
SAVE_THRESHOLD        = 150        # save the agent when reward exceeds this
SAVE_DIR              = os.path.expanduser('~/storm_dqn/storm_agents')
LOG_DIR               = os.path.expanduser('~/storm_dqn/logs')

N_OBS     = 50
N_ACTIONS = 11


# ---------------------------------------------------------------------------
# Q-Network — identical to the paper: 50 -> 300 ReLU -> 300 ReLU -> 11
# ---------------------------------------------------------------------------
def build_qnet(name='QNet'):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_OBS,)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(N_ACTIONS, activation='linear'),
    ], name=name)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, maxlen):
        self.buf = collections.deque(maxlen=maxlen)

    def add(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, float(done)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        obs, act, rew, nobs, done = zip(*batch)
        return (np.array(obs,  dtype=np.float32),
                np.array(act,  dtype=np.int32),
                np.array(rew,  dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# DDQN training step
# Double DQN: q_net selects the action, target_net evaluates it.
# ---------------------------------------------------------------------------
@tf.function
def train_step(q_net, target_net, optimizer,
               obs_b, act_b, rew_b, nobs_b, done_b):
    # Best action according to q_net (Double DQN)
    best_actions = tf.argmax(q_net(nobs_b, training=False), axis=1)
    # Value of that action according to target_net
    nq = tf.reduce_sum(
        target_net(nobs_b, training=False) *
        tf.one_hot(best_actions, N_ACTIONS), axis=1)
    # Bellman target
    targets = rew_b + GAMMA * nq * (1.0 - done_b)

    with tf.GradientTape() as tape:
        q_pred = tf.reduce_sum(
            q_net(obs_b, training=True) *
            tf.one_hot(act_b, N_ACTIONS), axis=1)
        loss = tf.reduce_mean(tf.square(targets - q_pred))

    grads = tape.gradient(loss, q_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
    return loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    BETA = 0.997

    # Unique name for this run, e.g. "20260101-120000_beta_0.997_training_map"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_beta_{BETA}_training_map"

    # Full path of the run's log subfolder
    log_dir = os.path.join(os.path.expanduser('~/storm_dqn/logs'), run_name)
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard writer for this specific run
    writer = tf.summary.create_file_writer(log_dir)

    print(f"--- Starting training: {run_name} ---")
    print(f"--- Logs saved in: {log_dir} ---")

    # Override the environment reward values with the paper's before creating
    # the env (storm_env defaults may differ).
    import storm_env
    storm_env.REWARD_STEP      = REWARD_STEP
    storm_env.REWARD_COLLISION = REWARD_COLLISION

    print('Creating environment...')
    env = StormEnv()
    print('Environment ready.\n')

    # Networks
    q_net      = build_qnet('online')
    target_net = build_qnet('target')
    target_net.set_weights(q_net.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    buffer    = ReplayBuffer(BUFFER_SIZE)

    # Training state
    best_avg = -np.inf
    epsilon     = EPSILON_START
    total_steps = 0
    ep_rewards  = []
    t_start     = time.time()

    print('=' * 60)
    print('DDQN Training — Feng et al. (2021)')
    print(f'Episodes: {MAX_EPISODES} | Buffer: {BUFFER_SIZE} | Batch: {BATCH_SIZE}')
    print(f'LR: {LR} | Gamma: {GAMMA} | Target update: every {TARGET_UPDATE_FREQ} steps')
    print(f'Epsilon decay: multiplicative beta={BETA} per episode')
    print(f'  epsilon: {EPSILON_START} -> {EPSILON_MIN} '
          f'(after ~{int(np.log(EPSILON_MIN)/np.log(BETA))} episodes)')
    print(f'Reward: +{REWARD_STEP} (step) / {REWARD_COLLISION} (collision)')
    print(f'TensorBoard: tensorboard --logdir {LOG_DIR}')
    print('=' * 60)

    for ep in range(1, MAX_EPISODES + 1):

        obs, _ = env.reset()
        ep_reward = 0
        ep_steps  = 0
        losses    = []

        for _ in range(MAX_STEPS_PER_EPISODE):

            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, N_ACTIONS - 1)
            else:
                qvals  = q_net(obs[np.newaxis], training=False).numpy()[0]
                action = int(np.argmax(qvals))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, reward, next_obs, done)
            obs        = next_obs
            ep_reward += reward
            ep_steps  += 1
            total_steps += 1

            # Train only once the buffer is large enough
            if len(buffer) >= BATCH_SIZE:
                obs_b, act_b, rew_b, nobs_b, done_b = buffer.sample(BATCH_SIZE)
                loss = train_step(
                    q_net, target_net, optimizer,
                    obs_b, act_b, rew_b, nobs_b, done_b)
                losses.append(float(loss))

                # Update the target network every TARGET_UPDATE_FREQ steps
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    target_net.set_weights(q_net.get_weights())

            if done:
                break

        # --- End of episode ---

        # Multiplicative per-episode epsilon decay — paper Eq. (3): e_{k+1} = beta * e_k
        if epsilon > EPSILON_MIN:
            epsilon = max(EPSILON_MIN, epsilon * BETA)

        ep_rewards.append(ep_reward)
        avg      = np.mean(ep_rewards[-WINDOW_SIZE:])
        avg_loss = float(np.mean(losses)) if losses else 0.0
        elapsed  = (time.time() - t_start) / 60.0

        # --- Best checkpoint by moving average (avg50) ---
        # Save only once the window is full (ep >= WINDOW_SIZE); otherwise the
        # first few episodes skew the average and you save noise.
        if ep >= WINDOW_SIZE and avg > best_avg:
            best_avg = avg
            path = os.path.join(SAVE_DIR, 'storm_ddqn_best.weights.h5')
            q_net.save_weights(path)
            print(f'  *** NEW BEST avg50={avg:.1f} (ep {ep}) -> {path}')

        # TensorBoard logging
        with writer.as_default():
            tf.summary.scalar('reward/episode',    ep_reward,   step=ep)
            tf.summary.scalar('reward/avg50',      avg,         step=ep)
            tf.summary.scalar('train/loss',        avg_loss,    step=ep)
            tf.summary.scalar('train/epsilon',     epsilon,     step=ep)
            tf.summary.scalar('train/total_steps', total_steps, step=ep)

        # Terminal output
        print(f'Ep {ep:4d}/{MAX_EPISODES}  '
              f'steps {ep_steps:4d}  '
              f'reward {ep_reward:9.1f}  '
              f'avg50 {avg:9.1f}  '
              f'loss {avg_loss:.4f}  '
              f'eps {epsilon:.4f}  '
              f'{elapsed:.1f}min')

        # Save whenever the reward exceeds the threshold
        if ep_reward >= SAVE_THRESHOLD:
            path = os.path.join(SAVE_DIR, f'agent_ep{ep}_r{int(ep_reward)}.weights.h5')
            q_net.save_weights(path)
            print(f'  *** Saved: {path}')

        # Checkpoint every 100 episodes
        if ep % 100 == 0:
            path = os.path.join(SAVE_DIR, f'checkpoint_ep{ep}.weights.h5')
            q_net.save_weights(path)
            print(f'  [checkpoint] {path}')

    # Final agent
    final = os.path.join(SAVE_DIR, 'storm_ddqn_final.weights.h5')
    q_net.save_weights(final)
    print(f'Best avg50 reached: {best_avg:.1f}')
    print(f'Best checkpoint: {os.path.join(SAVE_DIR, "storm_ddqn_best.weights.h5")}')
    print(f'\nTraining complete. Final agent: {final}')
    env.close()


if __name__ == '__main__':
    main()
