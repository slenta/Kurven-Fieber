import torch
import pygame
import config as cfg
from curveai.nn import DQN
from curveai.ai_player import init_ai_player
from utils.game_functions import simulation_game_loop


# Load Ai_models
players = []
num_players = cfg.num_ai_players
for id in range(cfg.num_ai_players):
    model = DQN(num_players)
    if cfg.resume_iter != 0:
        model.load_state_dict(
            torch.load(
                f"{cfg.save_model_path}model_checkpoint_iter_{cfg.resume_iter}_player_{id}.pth"
            )
        )
    player = init_ai_player(id, model, cfg.resume_iter)
    players.append(player)


# Initialize remaining variables
last_spawn_time = pygame.time.get_ticks()
win_counts = {i + 1: 0 for i in range(num_players)}

# Call simulation game function
simulation_game_loop(
    players, num_players, win_counts, last_spawn_time, iteration=cfg.resume_iter
)
