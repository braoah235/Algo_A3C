# Test Agent

import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic


# no exploration in the testing phase
def test(rank, params, shared_model, model_type="standard"):
    torch.manual_seed(params.seed + rank)  # asynchronizing the test agents
    env = create_atari_env(
        params.env_name,
        video=True,
        stack_frames=params.stack_frames,
        clip_rewards=False,
        episodic_life=False,
    )
    env.seed(params.seed + rank)  # asynchronizing the environment
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.eval()  # set in eval mode as it is not training anymore

    state = env.reset()
    state = torch.from_numpy(state).float()
    reward_sum = 0
    done = True
    start_time = time.time()
    actions = deque(maxlen=100)
    episode_length = 0

    recent_rewards = deque(maxlen=max(1, int(getattr(params, "test_moving_avg_window", 20))))
    best_avg_reward = float("-inf")

    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())

        with torch.no_grad():
            _value, action_value = model(state.unsqueeze(0))
            prob = F.softmax(action_value, dim=1)
            action = prob.max(1)[1].item()  # greedy action in testing

        state, reward, done, _ = env.step(int(action))
        reward_sum += reward

        if done:
            print(
                "Time {}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    reward_sum,
                    episode_length,
                )
            )

            recent_rewards.append(float(reward_sum))
            if len(recent_rewards) >= recent_rewards.maxlen:
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_path = getattr(params, "best_model_path", f"Model/{model_type}_best.pth")
                    torch.save(shared_model.state_dict(), best_path)
                    print(
                        "New best moving avg reward {:.3f} over {} episodes -> {}".format(
                            avg_reward, len(recent_rewards), best_path
                        )
                    )

            # Save test results
            with open("results_standard.txt", "a") as f:
                f.write(f"{time.time()},{reward_sum},{episode_length}\n")

            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)  # one min sleep to let train agents progress

        state = torch.from_numpy(state).float()
