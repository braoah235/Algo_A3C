# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model_dueling import ActorCritic   # 👈 IMPORTANT

# Implementing a function to make sure the models share the same gradient
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if param.grad is not None:
            shared_param.grad = param.grad.clone() 


def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank)

    env = create_atari_env(
        params.env_name,
        stack_frames=params.stack_frames,
        clip_rewards=True,
        episodic_life=True,
    )
    env.seed(params.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    state = env.reset()
    state = torch.from_numpy(state).float()
    done = True
    train_steps = 0
    prev_lives = None

    episode_length = 0

    while True:

        model.load_state_dict(shared_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(params.num_steps):
            episode_length += 1

            # forward (dueling compatible)
            value, policy_logits = model(state.unsqueeze(0))
            if (not torch.isfinite(value).all()) or (not torch.isfinite(policy_logits).all()):
                # Recover from occasional numeric blow-ups in async training.
                model.load_state_dict(shared_model.state_dict())
                state = env.reset()
                state = torch.from_numpy(state).float()
                done = True
                prev_lives = None
                break

            prob = F.softmax(policy_logits, dim=1)
            log_prob = F.log_softmax(policy_logits, dim=1)

            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()

            log_prob = log_prob.gather(1, action)
            log_probs.append(log_prob)

            values.append(value)

            state, reward, done, info = env.step(int(action.item()))
            train_steps += 1

            done = (done or episode_length >= params.max_episode_length)

            shaped_reward = reward + params.step_penalty
            lives = info.get("lives") if isinstance(info, dict) else None
            if (
                lives is not None
                and prev_lives is not None
                and lives < prev_lives
            ):
                shaped_reward += params.life_loss_penalty
            if lives is not None:
                prev_lives = lives

            if params.clip_reward:
                shaped_reward = max(min(shaped_reward, 1), -1)

            if done:
                episode_length = 0
                state = env.reset()
                prev_lives = None

            state = torch.from_numpy(state).float()
            rewards.append(shaped_reward)

            if done:
                break

        #  bootstrap
        R = torch.zeros(1, 1)

        if not done:
            with torch.no_grad():
                value, _ = model(state.unsqueeze(0))
            R = value.detach()

        values.append(R)

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        gae_terms = []
        log_prob_terms = []
        entropy_terms = []

        for i in reversed(range(len(rewards))):

            R = params.gamma * R + rewards[i]

            advantage = R - values[i]
            value_loss += 0.5 * advantage.pow(2)

            TD = rewards[i] + params.gamma * values[i+1].detach() - values[i].detach()
            gae = gae * params.gamma * params.tau + TD

            gae_terms.append(gae)
            log_prob_terms.append(log_probs[i])
            entropy_terms.append(entropies[i])

        entropy_progress = min(1.0, train_steps / max(1, params.entropy_decay_steps))
        entropy_coef = params.entropy_coef_start + entropy_progress * (params.entropy_coef_end - params.entropy_coef_start)

        if gae_terms:
            gae_tensor = torch.stack(gae_terms)
            if gae_tensor.numel() > 1:
                gae_std = gae_tensor.std(unbiased=False)
                gae_tensor = (gae_tensor - gae_tensor.mean()) / (gae_std + 1e-8)
            else:
                gae_tensor = gae_tensor - gae_tensor.mean()
            for i in range(len(gae_terms)):
                policy_loss += -log_prob_terms[i] * gae_tensor[i].detach() - entropy_coef * entropy_terms[i]

        optimizer.zero_grad()

        loss = policy_loss + 0.5 * value_loss
        if not torch.isfinite(loss).all():
            optimizer.zero_grad()
            continue
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
