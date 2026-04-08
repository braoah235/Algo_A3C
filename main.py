# Main code

from __future__ import print_function
import os
import torch
import torch.multiprocessing as mp
from envs import create_atari_env
import sys
import my_optim

# choix du modele
MODEL_TYPE = "standard"  # "standard" ou "dueling"
RESUME_PATH = None

if len(sys.argv) > 1:
    MODEL_TYPE = sys.argv[1].strip().lower()
if len(sys.argv) > 2:
    RESUME_PATH = sys.argv[2].strip()

if MODEL_TYPE == "standard":
    from model import ActorCritic
    from train import train       # train standard
    from test import test         # test standard
elif MODEL_TYPE == "dueling":
    from model_dueling import ActorCritic
    from train_dueling import train  # train dueling
    from test_dueling import test    # test dueling
else:
    raise ValueError(
        "MODEL_TYPE invalide. Utilise: 'standard' ou 'dueling'. "
        "Exemples: python main.py | python main.py dueling"
    )


# Gathering all the parameters (that we can modify to explore)
class Params():
    def __init__(self):
        self.lr = 0.00005
        self.gamma = 0.995
        self.tau = 1.0
        self.seed = 1
        self.num_processes = 19
        self.num_steps = 50
        self.max_episode_length = 10000
        self.env_name = 'Breakout-v0'
        self.stack_frames = 4
        self.clip_reward = True
        # Keep Atari reward shaping neutral by default; strong penalties can stall learning.
        self.step_penalty = 0.0
        self.life_loss_penalty = 0.0
        self.entropy_coef_start = 0.02
        self.entropy_coef_end = 0.005
        self.entropy_decay_steps = 800000
        self.resume_path = RESUME_PATH
        self.test_moving_avg_window = 20
        self.best_model_path = None


def main():
    # Main run
    os.environ['OMP_NUM_THREADS'] = '1'
    params = Params()
    torch.manual_seed(params.seed)

    env = create_atari_env(params.env_name, stack_frames=params.stack_frames)

    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()
    params.best_model_path = f"Model/{MODEL_TYPE}_best.pth"

    # Resume policy: explicit path > auto-best checkpoint.
    resume_candidate = params.resume_path
    if not resume_candidate:
        auto_best = params.best_model_path
        if os.path.exists(auto_best):
            resume_candidate = auto_best
    if resume_candidate:
        if os.path.exists(resume_candidate):
            checkpoint = torch.load(resume_candidate, map_location="cpu")
            shared_model.load_state_dict(checkpoint)
            print(f"Checkpoint chargé: {resume_candidate}")
        else:
            print(f"Checkpoint introuvable: {resume_candidate}. Démarrage depuis zéro.")

    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr)
    optimizer.share_memory()

    processes = []

    # test process
    p = mp.Process(target=test, args=(params.num_processes, params, shared_model, MODEL_TYPE))
    p.start()
    processes.append(p)

    # train processes
    for rank in range(0, params.num_processes):
        p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
        p.start()
        processes.append(p)

    # Sauvegarde toutes les 1 heure
    import time
    save_interval = 3600  # 1 heure
    save_count = 0
    while True:
        time.sleep(save_interval)
        save_count += 1
        model_path = f'Model/{MODEL_TYPE}_model_hour_{save_count}.pth'
        torch.save(shared_model.state_dict(), model_path)
        print(f'Modèle sauvegardé : {model_path} après {save_count} heure(s)')

    for p in processes:
        p.join()


if __name__ == '__main__':
    mp.freeze_support()
    main()
