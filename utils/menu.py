import pygame
from utils.get_keys import get_player_keys
import config as cfg

cfg.set_args()


def show_menu_screen(color=cfg.white, font=cfg.font):

    # Create screen
    screen = pygame.display.set_mode((cfg.screen_width, cfg.screen_height))
    pygame.display.set_caption("Curve Fever")

    # Show menu screen to select number of players
    num_players = 0
    while num_players not in [1, 2, 3, 4, 5, 6]:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    num_players = 1
                elif event.key == pygame.K_2:
                    num_players = 2
                elif event.key == pygame.K_3:
                    num_players = 3
                elif event.key == pygame.K_4:
                    num_players = 4
                elif event.key == pygame.K_5:
                    num_players = 5
                elif event.key == pygame.K_6:
                    num_players = 6

        # Draw menu screen
        screen.fill(color)
        text = font.render("Select number of players (1-6): ", True, (0, 0, 0))
        screen.blit(
            text,
            (
                cfg.screen_width / 2 - text.get_width() / 2,
                cfg.screen_height / 2 - text.get_height() / 2 - 50,
            ),
        )

        text = font.render("1 player", True, (0, 0, 0))
        screen.blit(
            text,
            (
                cfg.screen_width / 2 - text.get_width() / 2,
                cfg.screen_height / 2 - text.get_height() / 2 + 20,
            ),
        )
        text = font.render("2 players", True, (0, 0, 0))
        screen.blit(
            text,
            (
                cfg.screen_width / 2 - text.get_width() / 2,
                cfg.screen_height / 2 - text.get_height() / 2 + 50,
            ),
        )
        text = font.render("3 players", True, (0, 0, 0))
        screen.blit(
            text,
            (
                cfg.screen_width / 2 - text.get_width() / 2,
                cfg.screen_height / 2 - text.get_height() / 2 + 80,
            ),
        )
        text = font.render("4 players", True, (0, 0, 0))
        screen.blit(
            text,
            (
                cfg.screen_width / 2 - text.get_width() / 2,
                cfg.screen_height / 2 - text.get_height() / 2 + 110,
            ),
        )
        text = font.render("5 players", True, (0, 0, 0))
        screen.blit(
            text,
            (
                cfg.screen_width / 2 - text.get_width() / 2,
                cfg.screen_height / 2 - text.get_height() / 2 + 140,
            ),
        )
        text = font.render("6 players", True, (0, 0, 0))
        screen.blit(
            text,
            (
                cfg.screen_width / 2 - text.get_width() / 2,
                cfg.screen_height / 2 - text.get_height() / 2 + 170,
            ),
        )

        pygame.display.update()

    # Prompt user to enter keys for each player
    player_keys = get_player_keys(num_players, screen, color=cfg.white, font=cfg.font)

    return num_players, player_keys
