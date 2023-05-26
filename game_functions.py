import pygame
from player_functions import get_random_position, get_random_gap
from player_functions import move_players, get_winner
from display_functions import display_text, draw_players, display_winner, draw_screen
from items import render_items, power_down
import config as cfg


def reset_game(players, screen, player_keys, win_counts):
    # Set all players to be alive and reset their positions and directions
    gap = False
    for player in players:
        player["alive"] = True
        start_pos, start_dir = get_random_position()
        start_gap, start_line = get_random_gap()
        player["speed"] = cfg.speed
        player["left"] = player_keys[player["id"]]["left"]
        player["right"] = player_keys[player["id"]]["right"]
        player["pos"] = start_pos
        player["dir"] = start_dir
        player["gap_timer"] = start_gap
        player["line_timer"] = start_line
        player["pos_history"] = [start_pos]
        player["del_angle"] = 5
        player["gap"] = gap
        player["gap_history"] = [gap]
        player["items"] = []
        player["item_timer"] = []

    # Reset winner
    winner = None
    items = []

    # Refresh the screen
    draw_screen(screen, win_counts)
    pygame.display.flip()

    # Pause briefly to give players time to reposition
    pygame.time.wait(1000)

    return winner, items


def game_loop(players, num_players, player_keys, screen, win_counts, last_spawn_time):
    # set up game variables
    game_over = False
    running = True
    winner = None
    items = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and game_over:
                    winner, items = reset_game(players, screen, player_keys, win_counts)
                    game_over = False
                elif event.key == pygame.K_ESCAPE and game_over:
                    running = False

        # Get current time in milliseconds
        current_time = pygame.time.get_ticks()

        # Check if it's time to spawn a new item (every 10 seconds)
        if current_time - last_spawn_time >= 1000:
            items = render_items(screen, items)

            # Update the last spawn time
            last_spawn_time = current_time

        # move all players and check for collisions
        move_players(players, items)

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
                        for player in players:
                            for i in range(len(player["items"])):
                                player, player_key = power_down(
                                    player["items"][i],
                                    player["item_timer"][i],
                                    player,
                                    player_key,
                                )

                        winner, items = reset_game(
                            players, num_players, player_keys, win_counts
                        )
                    elif event.key == pygame.K_ESCAPE:
                        running = False

        # Draw players
        draw_players(players, screen)

        pygame.display.update()
        cfg.clock.tick(60)

    pygame.quit()
