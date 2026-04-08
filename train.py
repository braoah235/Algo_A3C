# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic

# Implementing a function to make sure the models share the same gradient
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if param.grad is not None:
            shared_param.grad = param.grad.clone() 


def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank)   #shifting the seed with rank to asynchronize each training agent
    env = create_atari_env(
        params.env_name,
        stack_frames=params.stack_frames,
    ) #breakout-V0
    env.seed(params.seed + rank) #aligning the seed of the environment on the seed of the agent
    model = ActorCritic(env.observation_space.shape[0], env.action_space) 
    state = env.reset() #state is the inp image, which is a np array of 1*42*42 (grayscale)
    state = torch.from_numpy(state).float()
    done = True #to indicate when the game is done
    train_steps = 0
    prev_lives = None
    
    episode_length = 0
    while True:
        model.load_state_dict(shared_model.state_dict()) #synchronizing with the shared model - the agent gets the shared model to do an exploration on num_steps
        if done:
            pass
            
        else:
            pass
            
        #values to be calc in each step as exploration goes on
        values = [] #V(s)
        log_probs = []
        rewards = []
        entropies = []
        
        for step in range(params.num_steps):
            episode_length += 1
            #going through the num_steps exploration steps
            value, action_values = model(state.unsqueeze(0))
            if (not torch.isfinite(value).all()) or (not torch.isfinite(action_values).all()):
                # Recover from occasional numeric blow-ups in async training.
                model.load_state_dict(shared_model.state_dict())
                state = env.reset()
                state = torch.from_numpy(state).float()
                done = True
                prev_lives = None
                break
            prob = F.softmax(action_values, dim=1)
            log_prob = F.log_softmax(action_values, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial(num_samples=1).detach() #selecting a random draw from the distribution obtained
            log_prob = log_prob.gather(1, action) #get the log_prob correspoding to the action chosen
            log_probs.append(log_prob)
            values.append(value)
            state, reward, done, info = env.step(int(action.item()))
            train_steps += 1
            done = (done or episode_length >= params.max_episode_length) #done if the agent gets stuck for too long

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
                shaped_reward = max(min(shaped_reward, 1), -1) #clip between -1 and 1

            if done:
                episode_length = 0
                state = env.reset()
                prev_lives = None
            state = torch.from_numpy(state).float()
            rewards.append(shaped_reward)
            if done:
                break #we stop the exploration and we directly move on to the next step: the update of the shared model
        R = torch.zeros(1, 1) #cumulative reward init
        if not done:
            with torch.no_grad():
                value, _ = model(state.unsqueeze(0))
            R = value.detach() #value of the last reached state s is passed onto the cumReward
        values.append(R) #storing the value of the last reached state s
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1) #this is the generalized advantage estimation, which is A(a, s) = Q(s, a) - V(s)
        gae_terms = []
        log_prob_terms = []
        entropy_terms = []
        
        #now we start from the lasyt exploration and go back in time to update
        for i in reversed(range(len(rewards))):
            #we explore based on rewards only
            R = params.gamma * R + rewards[i]  # R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
                #here already in the first loop , R = V(last state)
            advantage = R - values[i]  # R is an estimator of Q at time t = i so advantage_i = Q_i - V(state_i) = R - value[i]  
            value_loss = value_loss + 0.5 * advantage.pow(2)
            #we need policy loss, so we need gae, for which we need Temporal diff TD
            TD = rewards[i] + params.gamma * values[i+1].detach() - values[i].detach()
            gae = gae * params.gamma * params.tau + TD #gae = sum_i (gamma*tau)^i * TD(i) with gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))
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
        loss = (policy_loss + 0.5 * value_loss)
        if not torch.isfinite(loss).all():
            optimizer.zero_grad()
            continue
        loss.backward() #as policy loss is smaller and we are trying to minimise, give 2x importance to that than value loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40) #clamping the values of gradient between 0 and 40 to prevent the gradient from taking huge values and degenerating the algorithm
        ensure_shared_grads(model, shared_model) #ensure that the agents' model and the shared model are using same grads
        optimizer.step()
