import numpy as np
import cv2
import math
import time
import yaml
import pygame


def generateAgents(spawn_type):
    # agent: y, x, angle
    # agents: y, x, angle, species
    agents = np.zeros((n_agents, 4), dtype=np.float32)

    center_y, center_x = gridsize[0] / 2, gridsize[1] / 2
    max_radius = min(gridsize[0], gridsize[1]) * 0.2  # circle radius

    # Random radii (uniform in area)
    if spawn_type == "no_fill":
        r = max_radius
    elif spawn_type == "fill":
        r = max_radius * np.sqrt(rng.random(n_agents))  # sqrt for uniform area distribution
    theta = rng.random(n_agents) * 2 * math.pi       # random angles

    agents[:,0] = center_y + r * np.sin(theta)  # y position
    agents[:,1] = center_x + r * np.cos(theta)  # x position
    agents[:,2] = np.arctan2(center_y - agents[:,0], center_x - agents[:,1])
    agents[:,3] = rng.integers(0, n_species, size=n_agents)
    return agents

def sense(agents, angles, kernel_sum):
    dy, dx = np.sin(angles) * sensor_distance, np.cos(angles) * sensor_distance
    sense_positions = agents[:,:2].copy()
    sense_positions[:,0] += dy
    sense_positions[:,1] += dx
    sense_positions[:,0] = np.clip(sense_positions[:,0], 0, gridsize[0]-1)
    sense_positions[:,1] = np.clip(sense_positions[:,1], 0, gridsize[1]-1)
    sense_positions = sense_positions.astype(np.uint32)
    return kernel_sum[sense_positions[:,0], sense_positions[:,1]]

def updateAgents(agents, trails):
    angles = agents[:,2]

    kernel_sums = cv2.blur(trails, (sensor_size, sensor_size))
    if kernel_sums.ndim == 2:
        kernel_sums = kernel_sums.reshape((*kernel_sums.shape, n_species))

    total = np.sum(kernel_sums, axis=-1)
    for i in range(n_species):
        species_mask = agents[:,3] == i
        agents_masked = agents[species_mask]
        angles_masked = agents_masked[:,2]

        own = kernel_sums[..., i]
        other = total - own
        kernel_sum = own - other

        weights_forward = sense(agents_masked, angles_masked, kernel_sum)
        weights_left = sense(agents_masked, angles_masked - sensor_angle_spacing, kernel_sum)
        weights_right = sense(agents_masked, angles_masked + sensor_angle_spacing, kernel_sum)

        random_mask = (weights_forward < weights_left) & (weights_forward < weights_right)
        left_mask   = (weights_left > weights_forward) & (weights_left > weights_right)
        right_mask  = (weights_right > weights_forward) & (weights_right > weights_left)

        angles_masked[random_mask] += (rng.random(np.sum(random_mask)) - 0.5)
        angles_masked[left_mask]   -= turn_speed * rng.random(np.sum(left_mask))
        angles_masked[right_mask]  += turn_speed * rng.random(np.sum(right_mask))

        agents[species_mask, 2] = angles_masked

    angles = agents[:,2]

    dy, dx = np.sin(angles), np.cos(angles)
    agents[:,0] += dy
    agents[:,1] += dx

    agents_alt = np.zeros_like(agents)
    agents_alt[:,0] = np.clip(agents[:,0], 0, gridsize[0]-1)
    agents_alt[:,1] = np.clip(agents[:,1], 0, gridsize[1]-1)
    agents_alt[:,2] = rng.random(n_agents, dtype=np.float32) * 2 * math.pi
    agents_alt[:,3] = agents[:,3]

    mask = ((0 <= agents[:,0]) & \
            (agents[:,0] <= gridsize[0]-1) & \
            (0 <= agents[:,1]) & \
            (agents[:,1] <= gridsize[1]-1)).reshape(-1, 1)

    agents = np.where(
        mask,
        agents,
        agents_alt
    )

    return agents

def updateTrails(agents_int, trails):
    diffused_trail = cv2.blur(trails, (3,3)).astype(np.uint8)
    if diffused_trail.ndim == 2:
        diffused_trail = diffused_trail.reshape((*diffused_trail.shape, n_species))
    
    trails = (1-diffusion_speed) * trails + diffusion_speed * diffused_trail
    mask = trails >= evaporation_speed
    trails[mask] -= evaporation_speed
    trails[~mask] = 0

    for i in range(n_species):
        species = agents_int[np.where(agents_int[:,3]==i)]
        trails[species[:,0], species[:,1], i] = 255
    #trails = trails.astype(np.uint8)
    return trails

def getDisplayGrid(agents_int, trails):
    h, w, n_species = trails.shape
    grid = np.zeros((h, w, 3), dtype=np.float32)

    trails_float = trails.astype(np.float32)

    # Compute display trails: own - sum of other species
    total = np.sum(trails_float, axis=-1, keepdims=True)  # shape (h, w, 1)
    display_trails = np.maximum(0, 2*trails_float - total)  # shape (h, w, n_species)

    # Blend each species color
    for i in range(n_species):
        color = np.array(trail_color[i], dtype=np.float32) / 255.0
        grid += display_trails[..., i:i+1] * color  # broadcast to RGB

    # Clip and convert to uint8
    grid = np.clip(grid, 0, 255).astype(np.uint8)

    # Optionally draw agents on top
    for i in range(n_species):
        species_agents = agents_int[agents_int[:,3] == i]
        grid[species_agents[:,0], species_agents[:,1]] = trail_color[i]

    # Resize for display
    scale = min(1200/gridsize[0], 1980/gridsize[1]) / 1.25
    grid = cv2.resize(grid, dsize=None, interpolation=cv2.INTER_LINEAR, fx=scale, fy=scale)
    # trail colors in config.yaml are BGR (authored for OpenCV) -> flip to RGB for pygame
    return grid[..., ::-1]


with open("config.yaml", "r") as f:
    configs = yaml.safe_load(f)["configs"]


def load_config(config_idx):
    global trail_color, n_agents, n_species, gridsize, sensor_distance, sensor_size, sensor_angle_spacing, turn_speed, diffusion_speed, evaporation_speed, spawn_type
    config = configs[config_idx]
    trail_color = config["trail_color"]
    n_agents = config["n_agents"]
    n_species = config["n_species"]

    # rebuild gridsize properly
    gridsize = (*config["gridsize"], n_species)

    sensor_distance = config["sensor_distance"]
    sensor_size = config["sensor_size"]
    sensor_angle_spacing = config["sensor_angle_spacing"]
    turn_speed = config["turn_speed"]
    diffusion_speed = config["diffusion_speed"]
    evaporation_speed = config["evaporation_speed"]
    
    spawn_type = config["spawn_type"]

def reset_sim():
    global agents, trails
    load_config(config_idx)
    agents = generateAgents(spawn_type=spawn_type)
    trails = np.zeros(gridsize, dtype=np.uint8)


config_idx = 0
rng = np.random.default_rng(seed=42)
reset_sim()

pygame.init()
window = None
clock = pygame.time.Clock()

start_time = time.time()
frames = 0
running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_DOWN:
                    reset_sim()
                elif event.key == pygame.K_LEFT:
                    config_idx = (config_idx - 1) % len(configs)
                    reset_sim()
                elif event.key == pygame.K_RIGHT:
                    config_idx = (config_idx + 1) % len(configs)
                    reset_sim()

        frames += 1

        agents = updateAgents(agents, trails)
        agents_int = agents.astype(np.uint32)
        trails = updateTrails(agents_int, trails)
        grid = getDisplayGrid(agents_int, trails)

        # pygame surfaces are (width, height); numpy image is (height, width)
        size = (grid.shape[1], grid.shape[0])
        if window is None or window.get_size() != size:
            window = pygame.display.set_mode(size)
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(grid.transpose(1, 0, 2)))
        window.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(50)
        if frames % 30 == 0:
            name = configs[config_idx].get("name", config_idx)
            pygame.display.set_caption(f"Slime Mold - {name} - {clock.get_fps():.1f} FPS")
except KeyboardInterrupt:
    pass
finally:
    total_time = time.time() - start_time
    fps = round(1/(total_time/frames), 2)
    print(f"fps: {fps}")
    pygame.quit()
