import pygame
from player_functions import init_players
from game_functions import game_loop
from menu import show_menu_screen
from display_functions import draw_screen
import config as cfg

# Initialize pygame
pygame.init()

# Show screen and define number of players
num_players, player_keys = show_menu_screen()

# Initialize players and items
players = init_players(num_players)
last_spawn_time = pygame.time.get_ticks()

# Initialize win counts dictionary
win_counts = {i + 1: 0 for i in range(num_players)}

# Create Game screen
screen = pygame.display.set_mode((cfg.screen_width, cfg.screen_height))
draw_screen(screen, win_counts, screen_color=cfg.black, outline_color=cfg.white)

# Call game loop
game_loop(players, player_keys, num_players, screen, win_counts, last_spawn_time)
