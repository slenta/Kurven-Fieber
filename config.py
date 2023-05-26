import argparse
import pygame

# Define all needed variables
# Define screen dimensions
screen_width = None
screen_height = None
score_section_width = None
play_screen_width = None
# Player arguments
player_size = None
speed = None
# Define colors
white = None
black = None
red = None
green = None
# Define font
font = None
clock = None
# Define gap, line min and max
min_gap = None
max_gap = None
min_line = None
max_line = None
# Define item variables
num_items = None
item_size = None
item_time = None
item_letters = None

pygame.init()


def set_args(arg_file=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--screen_width", type=int, default=740)
    arg_parser.add_argument("--screen_height", type=int, default=480)
    arg_parser.add_argument("--score_section_width", type=int, default=200)
    arg_parser.add_argument("--speed", type=int, default=2)
    arg_parser.add_argument("--player_size", type=int, default=7)
    arg_parser.add_argument("--min_gap", type=int, default=10)
    arg_parser.add_argument("--max_gap", type=int, default=15)
    arg_parser.add_argument("--min_line", type=int, default=15)
    arg_parser.add_argument("--max_line", type=int, default=200)
    arg_parser.add_argument("--num_items", type=int, default=5)
    arg_parser.add_argument("--item_size", type=int, default=15)
    arg_parser.add_argument("--item_time", type=int, default=200)

    args = arg_parser.parse_args()

    global screen_width
    global screen_height
    global score_section_width
    global play_screen_width
    global speed
    global player_size
    global white
    global black
    global green
    global red
    global font
    global clock
    global min_gap
    global max_gap
    global min_line
    global max_line
    global num_items
    global item_size
    global item_time
    global item_letters

    screen_width = args.screen_width
    screen_height = args.screen_height
    score_section_width = args.score_section_width
    play_screen_width = args.screen_width - args.score_section_width
    speed = args.speed
    player_size = args.player_size
    white = (255, 255, 255)
    black = (0, 0, 0)
    green = (0, 255, 0)
    red = (255, 0, 0)
    font = pygame.font.SysFont(None, 30)
    clock = pygame.time.Clock()
    min_gap = args.min_gap
    max_gap = args.max_gap
    min_line = args.min_line
    max_line = args.max_line
    num_items = args.num_items
    item_size = args.item_size
    item_time = args.item_time
    # item ids: 0: self slower, 1: self faster, 2: others slower, 3: others faster, 4: direction change, 5: angle change
    item_letters = ["->", "--->", "->", "--->", "<->", "°"]
