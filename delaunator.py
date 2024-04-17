import pygame
import sys
import random

pygame.init()

canvas_width = 800
canvas_height = 600

# Set up colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED=(255,0,0)

def generate_random_points(num_points):
    points = []
    for _ in range(num_points):
        x = random.randint(0, canvas_width)
        y = random.randint(0, canvas_height)
        points.append((x, y))
    return points


points = generate_random_points(100)
# Create the canvas surface
canvas = pygame.display.set_mode((canvas_width, canvas_height))
pygame.display.set_caption("Pygame Canvas")



# Main loop
while True:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    canvas.fill(WHITE)
    for point in points:
        pygame.draw.circle(canvas, RED, point, 4)



    pygame.display.update()