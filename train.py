import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
from copy import deepcopy
import torch.nn.functional as F

import math
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

SAVE_DIR = "/acrobot_grid"


def norm_state(s):
    s = np.array(s, dtype=np.float32).copy()
    if s.shape[0] >= 6:
        s[4] /= (4 * np.pi)
        s[5] /= (9 * np.pi)
    return s


def select_action_eps_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, Q.n_actions))

    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        return int(Q(state).argmax().item())


def to_tensor(x, dtype=np.float32):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=dtype)
    x = torch.from_numpy(x)
    return x


def compute_td_loss(Q, states, actions, td_target, weights=None):
    s = to_tensor(states)  # [B, S]
    a = to_tensor(actions, int).long()  # [B]

    Q_s = Q(s)
    Q_s_a = Q_s.gather(1, a.unsqueeze(1)).squeeze(1)

    td_error = Q_s_a - td_target

    td_losses = F.smooth_l1_loss(Q_s_a, td_target, reduction='none')
    w = torch.tensor(weights, dtype=torch.float32, device=td_losses.device)
    loss = (td_losses * w).mean()
    return loss, torch.abs(td_error).detach()


def eval_dqn(env_name, Q, n_episodes=10, seed=0):
    env = gym.make(env_name)
    rets = []
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        done, ep_return = False, 0.

        while not done:
            # set epsilon = 0 to make an agent act greedy
            s = norm_state(s)
            a = select_action_eps_greedy(Q, s, epsilon=0.)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_return += r
            s = s_next
        rets.append(ep_return)
    env.close()
    return float(np.mean(rets))


def sample_prioritized_batch(replay_buffer, n_samples, candidate_size=4096, alpha=0.6, beta=0.4):
    n = len(replay_buffer)
    k = min(candidate_size, n)
    cand = np.random.randint(0, n, size=k)
    priorities = np.array([replay_buffer[i][0] for i in cand], dtype=np.float32)
    priorities = (np.abs(priorities) + 1e-6) ** alpha
    s = priorities.sum()
    if not np.isfinite(s) or s <= 0:
        priorities = np.ones_like(priorities) / len(priorities)
    else:
        priorities /= s
    idx_in_cand = np.random.choice(k, size=n_samples, replace=n_samples > k, p=priorities)
    sample_probs = priorities[idx_in_cand]

    weights = (1.0 / (sample_probs + 1e-12)) ** beta
    weights = weights / weights.max()
    weights = weights.astype(np.float32)

    indices = cand[idx_in_cand]
    v = [replay_buffer[i] for i in indices]
    _, states, actions, rewards, next_states, terminateds, n_steps = zip(*v)

    batch = (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(terminateds, dtype=np.bool),
        np.array(n_steps, dtype=np.int64),
    )
    return batch, indices, weights


def update_batch(replay_buffer, indices, new_priority):
    """Updates batches with corresponding indices
    replacing their priority values."""
    for i, idx in enumerate(indices):
        replay_buffer[idx] = (new_priority[i],) + replay_buffer[idx][1:]


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        cur_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(cur_dim, h_dim))
            layers.append(nn.ReLU())
            cur_dim = h_dim
        self.back = nn.Sequential(*layers)

        self.value_head = NoisyLinear(cur_dim, 1)
        self.adv_head = NoisyLinear(cur_dim, output_dim)
        self.n_actions = output_dim

    def forward(self, x):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        z = self.back(x)
        v = self.value_head(z)  # [B, 1]
        a = self.adv_head(z)  # [B, A]
        q = v + (a - a.mean(dim=1, keepdim=True))  # [B, A]

        return q.squeeze(0) if squeeze else q

    def reset_noise(self):
        self.value_head.reset_noise()
        self.adv_head.reset_noise()


def create_dueling_network(input_dim, hidden_dims, output_dim):
    return DuelingDQN(input_dim, hidden_dims, output_dim)


class AcrobotDenseReward(gym.Wrapper):
    def __init__(self, env, scale=0.5):
        super().__init__(env)
        # scale = 0.5 guarantees that max reward is -1 + 0.5 = -0.5
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # obs = [cos(th1), sin(th1), cos(th2), sin(th2), th1_dot, th2_dot]
        cos_th1, sin_th1, cos_th2, sin_th2 = obs[0], obs[1], obs[2], obs[3]

        # height = -cos(th1) - cos(th1 + th2)
        # cos(th1 + th2) = cos_th1*cos_th2 - sin_th1*sin_th2
        height = -cos_th1 - (cos_th1 * cos_th2 - sin_th1 * sin_th2)

        # Height in [-2, 2]
        # Normalize into [0, 1]
        height_norm = (height + 2.0) / 4.0

        dense_bonus = self.scale * height_norm
        shaped_reward = float(reward) + dense_bonus

        return obs, shaped_reward, terminated, truncated, info


def push_nstep_transition(nstep_buf, replay_buffer, rb_pos, replay_buffer_size,
                          gamma, max_prio):
    """
    take first elem from nstep_buf and make n-step transition
    return new rb_pos
    """
    # (s, a, r, s_next, terminated, truncated)
    s0, a0, _, _, _, _ = nstep_buf[0]

    R = 0.0
    sN = None
    terminated_within = False
    n_used = 0

    for i, (s, a, r, s_next, terminated, truncated) in enumerate(nstep_buf):
        R += (gamma ** i) * float(r)
        sN = s_next
        n_used = i + 1

        if terminated:
            terminated_within = True
            break
        if truncated:
            break

    item = (max_prio, s0, a0, R, sN, terminated_within, n_used)

    if len(replay_buffer) < replay_buffer_size:
        replay_buffer.append(item)
    else:
        replay_buffer[rb_pos] = item
        rb_pos = (rb_pos + 1) % replay_buffer_size

    nstep_buf.popleft()
    return rb_pos


def compute_td_target_ddqn_nstep(Q, Q_slow, rewards, next_states, terminateds, n_steps, gamma=0.99):
    r = to_tensor(rewards)  # [B]
    s_next = to_tensor(next_states)  # [B, S]
    term = to_tensor(terminateds, bool)  # [B]
    n = to_tensor(n_steps, int).long()  # [B]

    with torch.no_grad():
        a = Q(s_next).argmax(dim=1)  # [B]
        q = Q_slow(s_next).gather(1, a.unsqueeze(1)).squeeze(1)  # [B]
        gam = (gamma ** n.float())  # [B]
        target = r + gam * q * (~term)
    return target


def run_ddqn_prioritized_rb(
        env_name="CartPole-v1",
        hidden_dims=(256, 256), lr=1e-3, gamma=0.99,
        total_max_steps=100_000,
        train_schedule=4, replay_buffer_size=400, batch_size=32,
        eval_schedule=1000, smooth_ret_window=1,
        tau=0.005, success_ret=200.,
        start_learn=10000, nstep=3, seed=None
):
    best_avg_return = -1e9
    last_avg_return = None
    best_state_dict = None
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    env = gym.make(env_name)
    if seed is not None:
        env.action_space.seed(seed)
        s, _ = env.reset(seed=seed)
    else:
        s, _ = env.reset()
    if env_name == "Acrobot-v1":  # action repeat
        env = AcrobotDenseReward(env, scale=0.5)
    replay_buffer = []
    rb_pos = 0
    nstep_buf = deque()

    eval_return_history = deque(maxlen=smooth_ret_window)

    Q = create_dueling_network(
        input_dim=env.observation_space.shape[0],
        hidden_dims=hidden_dims,
        output_dim=env.action_space.n
    )
    opt = torch.optim.Adam(Q.parameters(), lr=lr)

    Q_slow = deepcopy(Q)

    s = norm_state(s)
    global_step = 1
    max_prio = 1.0
    Q.train()
    for global_step in range(1, total_max_steps + 1):
        beta = 0.4 + (1.0 - 0.4) * min(1.0, global_step / total_max_steps)
        if global_step < start_learn:
            a = env.action_space.sample()
            epsilon = 1.0  # for logs
        else:
            Q.reset_noise()
            a = select_action_eps_greedy(Q, s, epsilon=0.0)
            epsilon = 0.0
        s_next, r, terminated, truncated, _ = env.step(a)
        s_next = norm_state(s_next)
        done = terminated or truncated
        nstep_buf.append((s, a, r, s_next, terminated, truncated))
        # update after n steps
        if len(nstep_buf) >= nstep:
            rb_pos = push_nstep_transition(
                nstep_buf, replay_buffer, rb_pos, replay_buffer_size,
                gamma, max_prio
            )

        # update end states
        if terminated or truncated:
            while len(nstep_buf) > 0:
                rb_pos = push_nstep_transition(
                    nstep_buf, replay_buffer, rb_pos, replay_buffer_size,
                    gamma, max_prio
                )

        if global_step >= start_learn and global_step % train_schedule == 0:
            train_batch, indices, weights = sample_prioritized_batch(replay_buffer, batch_size, beta=beta)
            (states, actions, rewards, next_states, terminateds, n_steps) = train_batch

            opt.zero_grad()
            Q.reset_noise()
            Q_slow.reset_noise()
            td_target = compute_td_target_ddqn_nstep(Q, Q_slow, rewards, next_states, terminateds, gamma=gamma,
                                                     n_steps=n_steps)
            loss, td_losses = compute_td_loss(Q, states, actions, td_target, weights=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=10.0)
            opt.step()
            for target_param, local_param in zip(Q_slow.parameters(), Q.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            update_batch(
                replay_buffer, indices, td_losses.detach().abs().cpu().numpy().reshape(-1)
            )
            batch_max = float(td_losses.detach().max().item())
            if batch_max > max_prio:
                max_prio = batch_max

        if global_step % eval_schedule == 0:
            Q.eval()
            eval_return = eval_dqn(env_name, Q)
            Q.train()
            eval_return_history.append(eval_return)
            avg_return = np.mean(eval_return_history)
            last_avg_return = avg_return

            if avg_return > best_avg_return:
                best_avg_return = avg_return
                best_state_dict = {k: v.detach().cpu().clone() for k, v in Q.state_dict().items()}
            print(f'{global_step=} | {avg_return=:.3f} | {epsilon=:.3f}')
            if avg_return >= success_ret:
                print('Решено!')
                break

        s = s_next
        if done:
            s, _ = env.reset()
            s = norm_state(s)
    return {
        "best_avg_return": float(best_avg_return),
        "last_avg_return": None if last_avg_return is None else float(last_avg_return),
        "steps": int(global_step),
        "best_state_dict": best_state_dict,
        "eval_history": list(eval_return_history)
    }


BEST_CFG = dict(
    env_name="Acrobot-v1",
    hidden_dims=(256, 256),
    lr=3e-4,
    gamma=0.99,
    replay_buffer_size=100_000,
    batch_size=256,
    start_learn=10_000,
    eval_schedule=5_000,
    total_max_steps=1_000_000,
    nstep=1,
    train_schedule=2,
    tau=0.001,
    seed=4242
)

print("Starting training with optimal Rainbow-lite config...")
results = run_ddqn_prioritized_rb(**BEST_CFG)

print("\nTraining Finished!")
print(f"Best Evaluation Return: {results['best_avg_return']:.2f}")

if "best_state_dict" in results:
    _tmp_env = gym.make(BEST_CFG["env_name"])
    Q_final = create_dueling_network(
        input_dim=_tmp_env.observation_space.shape[0],
        hidden_dims=BEST_CFG["hidden_dims"],
        output_dim=_tmp_env.action_space.n
    )
    _tmp_env.close()

    Q_final.load_state_dict(results["best_state_dict"])
    Q_final.eval()

    final_score = eval_dqn(BEST_CFG["env_name"], Q_final, n_episodes=100, seed=1000)
    print(f"Final SOTA Evaluation (100 episodes): {final_score:.2f}")
if "eval_history" in results:
    history = results["eval_history"]
    eval_steps = np.arange(1, len(history) + 1) * BEST_CFG["eval_schedule"]

    plt.figure(figsize=(10, 5))
    plt.plot(eval_steps, history, label="Rainbow-lite DQN", color='blue', linewidth=2)
    plt.axhline(y=-62, color='red', linestyle='--', label='Theoretical Physical Limit (~ -62)')

    plt.title("Training Curve (Acrobot-v1)", fontsize=14)
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Evaluation Return (10 episodes avg)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=300)
    plt.show()
