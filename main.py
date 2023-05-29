import pygame
from nn import DQN
from player_functions import init_players
from game_functions import game_loop
from menu import show_menu_screen
from display_functions import draw_screen
from ai_player import init_ai_player
import config as cfg

# Initialize pygame
pygame.init()

# Show screen and define number of players
num_players, player_keys = show_menu_screen()

# Initialize players and items
players = []
for i in range(cfg.num_ai_players):
    iteration = cfg.resume_iter
    # if new model:
    model = DQN(num_players)
    # if loading old model
    # model = dqn(num_players)
    # model.load_state_dict(torch.load(f"{cfg.save_model_path}\model_checkpoint_iter_{iteration}_player_{i}.pth"))
    player = init_ai_player(i, model, iteration)
    players.append(player)
players = init_players(num_players - cfg.num_ai_players, player_keys, players)
last_spawn_time = pygame.time.get_ticks()

# Initialize win counts dictionary
win_counts = {i + 1: 0 for i in range(num_players)}

# Create Game screen
screen = pygame.display.set_mode((cfg.screen_width, cfg.screen_height))
draw_screen(screen, win_counts, screen_color=cfg.black, outline_color=cfg.white)

# Call game loop
game_loop(players, num_players, player_keys, screen, win_counts, last_spawn_time)
