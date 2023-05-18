import pygame
from player_functions import get_random_position, get_random_gap
from player_functions import move_players, get_winner
from display_functions import display_text, draw_players, display_winner, draw_screen
from items import render_items
import config as cfg


def reset_game(players, screen, win_counts):
    # Set all players to be alive and reset their positions and directions
    gap = False
    for player in players:
        player["alive"] = True
        start_pos, start_dir = get_random_position()
        start_gap, start_line = get_random_gap()
        player["pos"] = start_pos
        player["dir"] = start_dir
        player["gap_timer"] = start_gap
        player["line_timer"] = start_line
        player["pos_history"] = [start_pos]
        player["gap"] = gap
        player["gap_history"] = [gap]

    # for item in items:
    #     item["alive"] = False
    #     item["timer"] = 0

    # Reset winner
    winner = None

    # Refresh the screen
    draw_screen(screen, win_counts)
    pygame.display.flip()

    # Pause briefly to give players time to reposition
    pygame.time.wait(1000)

    return winner


def game_loop(players, player_keys, num_players, screen, win_counts):
    # set up game variables
    game_over = False
    running = True
    winner = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and game_over:
                    winner = reset_game(players, screen, win_counts)
                    game_over = False
                elif event.key == pygame.K_ESCAPE and game_over:
                    running = False

        # move all players and check for collisions
        move_players(players, player_keys)
        # render_items(screen, items)

        # check if any players are alive
        alive_players = [player for player in players if player["alive"]]
        if len(alive_players) <= 1:
            # Set game over to true
            game_over = True

            # Stop all players when the game is over
            for player in players:
                player["dir"] = "stop"

            # Get winning player
            if len(alive_players) == 1:
                if winner == None:
                    winner = get_winner(players, win_counts)
                    display_winner(screen, win_counts)

            # Display options to restart or quit
            display_text(
                screen,
                "Game Over",
                font=cfg.font,
                color=cfg.white,
                x=(cfg.screen_width - cfg.score_section_width) / 2,
                y=cfg.screen_height / 2 - 50,
            )
            display_text(
                screen,
                "Press Space to Restart",
                font=cfg.font,
                color=cfg.white,
                x=(cfg.screen_width - cfg.score_section_width) / 2,
                y=cfg.screen_height / 2 + 10,
            )
            display_text(
                screen,
                "Press Esc to Quit",
                font=cfg.font,
                color=cfg.white,
                x=(cfg.screen_width - cfg.score_section_width) / 2,
                y=cfg.screen_height / 2 + 50,
            )

            # Event handling for options
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        winner = reset_game(players, num_players, win_counts)
                    elif event.key == pygame.K_ESCAPE:
                        running = False

        # Draw players
        draw_players(players, screen)

        pygame.display.update()
        cfg.clock.tick(60)

    pygame.quit()
