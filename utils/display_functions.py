import pygame
import config as cfg


def display_text(screen, text, font, color, x, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (x, y)
    screen.blit(text_surface, text_rect)


def draw_screen(screen, win_counts, screen_color=cfg.black, outline_color=cfg.white):
    screen.fill(screen_color)  # Fill the screen with black color
    pygame.draw.rect(
        screen,
        cfg.white,
        (0, 0, cfg.screen_width - cfg.score_section_width, cfg.screen_height),
        5,
    )
    pygame.display.set_caption("Curve Fever")

    # Draw the score section
    pygame.draw.rect(
        screen,
        cfg.black,
        (
            cfg.screen_width - cfg.score_section_width,
            0,
            cfg.score_section_width,
            cfg.screen_height,
        ),
    )
    pygame.draw.rect(
        screen,
        cfg.white,
        (0, 0, cfg.screen_width - cfg.score_section_width, cfg.screen_height),
        5,
    )
    pygame.draw.rect(
        screen, outline_color, (0, 0, cfg.screen_width, cfg.screen_height), 5
    )

    # Display win counts
    for player_id, win_count in win_counts.items():
        player_text = f"Player {player_id}: {win_count} wins"
        display_text(
            screen,
            player_text,
            font=cfg.font,
            color=cfg.white,
            x=cfg.screen_width - cfg.score_section_width + 90,
            y=20 + player_id * 20,
        )

    return screen


def draw_players(players, screen):
    for player in players:
        pygame.draw.circle(screen, player["color"], player["pos"], cfg.player_size // 2)
        if player["gap"]:
            pygame.draw.circle(
                screen, cfg.black, player["pos_history"][1], cfg.player_size // 2
            )
        prev_pos = None
        for pos, gap in zip(player["pos_history"], player["gap_history"]):
            if gap:
                prev_pos = None
                continue
            if prev_pos is not None:
                pygame.draw.line(
                    screen,
                    player["color"],
                    prev_pos,
                    pos,
                    cfg.player_size,
                )
                prev_pos = pos


def display_winner(screen, win_counts):
    # Clear the score section
    pygame.draw.rect(
        screen,
        cfg.black,
        (
            cfg.screen_width - cfg.score_section_width,
            0,
            cfg.score_section_width,
            cfg.screen_height,
        ),
    )
    pygame.draw.rect(screen, cfg.white, (0, 0, cfg.screen_width, cfg.screen_height), 5)
    pygame.draw.rect(
        screen,
        cfg.white,
        (0, 0, cfg.screen_width - cfg.score_section_width, cfg.screen_height),
        5,
    )
    # Display win counts
    for player_id, win_count in win_counts.items():
        player_text = f"Player {player_id}: {win_count} wins"
        display_text(
            screen,
            player_text,
            font=cfg.font,
            color=cfg.white,
            x=cfg.screen_width - cfg.score_section_width + 90,
            y=20 + player_id * 20,
        )


def update_screen_with_game_state(game_state, screen):
    # Define colors for different game state values
    colors = {
        -1: (0, 0, 0),  # Black for walls
        0: (255, 255, 255),  # White for empty space
        1: (255, 0, 0),  # Red for old player points
        2: (0, 255, 0),  # Green for player 1
        3: (0, 0, 255),  # Blue for player 2
        4: (255, 255, 0),  # Yellow for player 3
        5: (0, 255, 255),  # Cyan for player 4
        # Add more colors if needed
    }

    # Iterate over each pixel in the game state
    for y in range(game_state.shape[0]):
        for x in range(game_state.shape[1]):
            value = int(game_state[y, x].item())  # Convert tensor to integer
            color = colors.get(value)
            if value >= 6:
                color = (255, 255, 0)
            if not color:
                print("no valid value:", value)
            screen.set_at((x, y), color)

    # Update the display
    pygame.display.flip()
