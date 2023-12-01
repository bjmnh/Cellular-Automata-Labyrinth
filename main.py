import pygame
import sys
import numpy as np
import random
import math
import enum


# Constants
WINDOW_SIZE = (1000, 900)
CELL_SIZE = 8  # Adjust to scale maze up or down
GRID_SIZE = (WINDOW_SIZE[0] // CELL_SIZE, WINDOW_SIZE[1] // CELL_SIZE)
END_POS = (GRID_SIZE[0] - 5, GRID_SIZE[1] // 2)
MIDDLE_POS = (GRID_SIZE[0] // 2, GRID_SIZE[1] // 2)
MEADOW_RADIUS = 3
START_POS = MIDDLE_POS
RULE_UPDATE_INTERVAL = 2000  # Apply cellular automaton rules every x milliseconds
WALL_REPLENISH_RATE = 0.411  # Probability of a wall being replenished in each update (requires fine balance)
PATH_PRINT = True  # Display the path

minotaur_image = pygame.image.load("minotaur.png")
goal_image = pygame.image.load("goal.png")
wall_image = pygame.image.load("pillar.png")
grass_image = pygame.image.load("grass.png")
mazewall_image = pygame.image.load("mazewall.png")
floor_image = pygame.image.load("floor.png")

minotaur_image = pygame.transform.scale(minotaur_image, (CELL_SIZE, CELL_SIZE))
goal_image = pygame.transform.scale(goal_image, (CELL_SIZE, CELL_SIZE))
wall_image = pygame.transform.scale(wall_image, (CELL_SIZE, CELL_SIZE))
grass_image = pygame.transform.scale(grass_image, (CELL_SIZE, CELL_SIZE))
mazewall_image = pygame.transform.scale(mazewall_image, (CELL_SIZE, CELL_SIZE))
floor_image = pygame.transform.scale(floor_image, (CELL_SIZE, CELL_SIZE))

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 102, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BROWN = (153, 76, 0)
WALL_COLOR = BROWN
SPECIAL_WALL_COLOR = BLACK

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Labyrinth and Minotaur Generator")
clock = pygame.time.Clock()

# Grid representing labyrinth
grid = np.zeros(GRID_SIZE, dtype=int)


class Direction(enum.Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


def distance(pos1, pos2):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return math.sqrt(dx * dx + dy * dy)


def count_neighboring_walls(grid, x, y):
    count = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE[0] and 0 <= ny < GRID_SIZE[1]:
                count += grid[nx, ny] == 1  # Only count normal walls
    return count


def apply_fixed_rings_automaton_rules(grid):
    new_grid = grid.copy()
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            if (x, y) != START_POS and distance((x, y), END_POS) > 3:
                walls = count_neighboring_walls(grid, x, y)
                dist = distance((x, y), MIDDLE_POS)
                # Preserve fixed walls
                if grid[x, y] == 2:
                    new_grid[x, y] = 2
                # Preserve the existing structure, including fixed walls
                elif dist <= MEADOW_RADIUS:
                    new_grid[x, y] = grid[x, y]
                # Apply the cellular automata rule
                else:
                    new_grid[x, y] = 1 if walls == 3 else 0
                # Replenish walls
                if new_grid[x, y] == 0 and grid[x, y] == 1:
                    if random.random() < WALL_REPLENISH_RATE:
                        new_grid[x, y] = 1
    grid[:] = new_grid


def generate_fixed_rings_labyrinth(grid):
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            # Exclude starting and ending positions
            if (x, y) != START_POS and (x, y) != END_POS:
                dist = distance((x, y), MIDDLE_POS)
                if dist <= MEADOW_RADIUS:  # Start Meadow
                    grid[x, y] = 3
                elif distance((x, y), END_POS) < 3:  # Goal Meadow
                    grid[x, y] = 3
                elif MEADOW_RADIUS * 7 < dist <= MEADOW_RADIUS * 7.8:  # Inner ring
                    if x < MIDDLE_POS[0] and y in range(
                        MIDDLE_POS[1] - 4, MIDDLE_POS[1] + 4
                    ):
                        grid[x, y] = 0
                    else:
                        grid[x, y] = 2
                elif MEADOW_RADIUS * 17 < dist <= MEADOW_RADIUS * 17.8:  # Outer ring
                    if x > MIDDLE_POS[0] and y in range(
                        MIDDLE_POS[1] - 7, MIDDLE_POS[1] + 7
                    ):
                        grid[x, y] = 0
                    else:
                        grid[x, y] = 2
                # Big Ring for Cell Size < 4
                # elif MEADOW_RADIUS * 30 < dist <= MEADOW_RADIUS * 31:
                #     if x > MIDDLE_POS[0] and y in range(
                #         MIDDLE_POS[1] - 8, MIDDLE_POS[1] + 8
                #     ):
                #         grid[x, y] = 0
                #     else:
                #         grid[x, y] = 2
                elif (
                    MEADOW_RADIUS * 17.5 < dist
                ):  # Adjust if adding big ring, Stops wall spawn outside of maze
                    grid[x, y] = 0
                else:
                    grid[x, y] = 0 if random.random() < 0.5 else 1


def find_optimal_path(grid, start_pos, end_pos):
    # Helper functions
    def get_neighbors(pos):
        x, y = pos
        neighbors = [
            (x + dx, y + dy)
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
            if 0 <= x + dx < GRID_SIZE[0]
            and 0 <= y + dy < GRID_SIZE[1]
            and grid[x + dx, y + dy] not in {1, 2}
        ]
        return neighbors

    def heuristic(pos):
        return distance(pos, end_pos)

    # Initialize data structures
    open_set = {start_pos}
    came_from = {}
    g_score = {start_pos: 0}
    f_score = {start_pos: heuristic(start_pos)}

    # A* pathfind algorithm
    while open_set:
        min_f_score = float("inf")
        current_pos = None
        for pos in open_set:
            if f_score[pos] < min_f_score:
                min_f_score = f_score[pos]
                current_pos = pos
        if current_pos == end_pos:
            # Construct path
            path = [current_pos]
            while path[-1] != start_pos:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        open_set.remove(current_pos)
        for neighbor_pos in get_neighbors(current_pos):
            tentative_g_score = g_score[current_pos] + 1
            if neighbor_pos not in g_score or tentative_g_score < g_score[neighbor_pos]:
                came_from[neighbor_pos] = current_pos
                g_score[neighbor_pos] = tentative_g_score
                f_score[neighbor_pos] = g_score[neighbor_pos] + heuristic(neighbor_pos)
                if neighbor_pos not in open_set:
                    open_set.add(neighbor_pos)

    # No path found
    return None


def move_minotaur(minotaur_pos, optimal_path, grid):
    if optimal_path and len(optimal_path) > 1:
        return optimal_path[1]
    else:
        # If there's no path, move randomly
        valid_directions = []
        for direction in Direction:
            dx, dy = direction.value
            new_x, new_y = minotaur_pos[0] + dx, minotaur_pos[1] + dy
            if (
                0 <= new_x < GRID_SIZE[0]
                and 0 <= new_y < GRID_SIZE[1]
                and grid[new_x, new_y] not in {1, 2}
            ):
                valid_directions.append((new_x, new_y))

        if valid_directions:
            return random.choice(valid_directions)
        else:
            return minotaur_pos


def draw_grid(screen, grid, optimal_path):
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            if grid[x, y] == 1:  # Normal Walls
                screen.blit(wall_image, (x * CELL_SIZE, y * CELL_SIZE))
            elif grid[x, y] == 2:  # Structured Walls
                screen.blit(mazewall_image, (x * CELL_SIZE, y * CELL_SIZE))
            elif grid[x, y] == 3:  # Meadow
                screen.blit(grass_image, (x * CELL_SIZE, y * CELL_SIZE))
            elif grid[x, y] == 0:  # Regular ground
                screen.blit(floor_image, (x * CELL_SIZE, y * CELL_SIZE))
            # Start position
            pygame.draw.rect(
                screen,
                GREEN,
                (
                    START_POS[0] * CELL_SIZE,
                    START_POS[1] * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ),
                0,
            )

    # Draw optimal path
    if PATH_PRINT and optimal_path:
        for path_pos in optimal_path[2:]:  # Skips the first path position
            pygame.draw.rect(
                screen,
                BLUE,  # You can choose another color for the path
                (
                    path_pos[0] * CELL_SIZE,
                    path_pos[1] * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ),
                1,
            )


def main():
    # Generate labyrinth
    generate_fixed_rings_labyrinth(grid)

    # Initialize Minotaur position
    minotaur_pos = START_POS
    # Main game loop
    last_rule_update_time = pygame.time.get_ticks()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current_time = pygame.time.get_ticks()
        if current_time - last_rule_update_time >= RULE_UPDATE_INTERVAL:
            apply_fixed_rings_automaton_rules(grid)
            last_rule_update_time = current_time

        optimal_path = find_optimal_path(grid, minotaur_pos, END_POS)

        # Ensure there is a path
        while optimal_path is None:
            apply_fixed_rings_automaton_rules(grid)
            optimal_path = find_optimal_path(grid, minotaur_pos, END_POS)

        minotaur_pos = move_minotaur(minotaur_pos, optimal_path, grid)

        # Draw grid
        screen.fill(DARK_GREEN)
        draw_grid(screen, grid, optimal_path)

        # Draw the goal image
        screen.blit(goal_image, (END_POS[0] * CELL_SIZE, END_POS[1] * CELL_SIZE))

        # Draw the minotaur image
        screen.blit(
            minotaur_image, (minotaur_pos[0] * CELL_SIZE, minotaur_pos[1] * CELL_SIZE)
        )
        pygame.display.flip()

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
