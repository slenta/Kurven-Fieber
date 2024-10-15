import pygame
import torch
import numpy as np
from curveai.nn import DQN
from utils.player_functions import init_players
from utils.game_functions import game_loop, simulation_game_loop
from utils.menu import show_menu_screen
from utils.display_functions import draw_screen
from curveai.ai_player import init_ai_player
import config as cfg

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Initialize pygame
pygame.init()

# Show screen and define number of players
num_players, player_keys = show_menu_screen()

# Initialize win counts dictionary, game state
win_counts = {i + 1: 0 for i in range(num_players)}
last_spawn_time = pygame.time.get_ticks()

# Initialize players and items
players = []
for i in range(cfg.num_ai_players):
    iteration = cfg.resume_iter
    # loading old model
    model = DQN(num_players)
    if cfg.resume_iter != 0:
        model.load_state_dict(
            torch.load(
                f"{cfg.save_model_path}model_checkpoint_iter_{cfg.resume_iter}_player_{i}.pth"
            )
        )
    player = init_ai_player(i, model, iteration)
    players.append(player)

if not cfg.training:
    players = init_players(num_players, player_keys, players)
    # Create Game screen
    screen = pygame.display.set_mode((cfg.screen_width, cfg.screen_height))
    print("screen created")
    draw_screen(screen, win_counts, screen_color=cfg.black, outline_color=cfg.white)
    print("screen drawn")
    # Call game loop
    game_loop(players, player_keys, screen, win_counts, last_spawn_time)
else:
    simulation_game_loop(players, win_counts, last_spawn_time, cfg.resume_iter)
