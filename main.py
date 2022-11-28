# =============================================================================================
# File:
#   main.py
#
# Author:
#   Rodrigo Muñoz Guerrero - A00572858
#
# Date:
#   27/11/2022
#
# Description:
#   This program simulates vacuum cleaners or "Roombas" moving randomly thorugh a space with
#   clean or dirty spaces which the cleaner must clean. It creates the simulation with Mesa,
#   a python's library to simulate multi-agent systems and matplotlib to grpahically show
#   the simulation and proces.
# =============================================================================================

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

import time
import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

import pandas as pd
import numpy as np

# --------------------------------------------- #
# ------------ CLASS OF ROOMBA AGENT ---------- #
# --------------------------------------------- #

class Roomba(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)
        self.movements = 0
        self.spacesCleaned = 0

    def step(self):

        # -- CHECKS IF FLOOR IS EMPTY -- #
        if self.model.floor[self.pos[0]][self.pos[1]] > 0:
            self.model.floor[self.pos[0]][self.pos[1]] -= 1
            self.spacesCleaned += 1

        else:

            # -- GETS ALL EMPTY ADJACENT CELLS -- #
            cells = []
            for cell in self.model.grid.iter_neighborhood((self.pos), moore = True, include_center = False):
                if self.model.grid.is_cell_empty((cell)):
                    cells.append(cell)
            
            # -- MOVES -- #
            if len(cells) > 0:
                moveTo = self.random.choice(cells)
                self.model.grid.move_agent(self, moveTo)
                self.movements += 1

def getGrid(model):
    grid = np.zeros((model.grid.width, model.grid.height))
    for (content, x, y) in model.grid.coord_iter():
        if model.grid.is_cell_empty((x, y)):
            if model.floor[x][y] == 0:
                grid[x][y] = 0
            elif model.floor[x][y] > 0:
                grid[x][y] = 1
        else:
            grid[x][y] = 7
    
    return grid

class House(Model):
    def __init__(self, width, height, numAgents, dirtPercentage):
        self.grid = MultiGrid(height, width, False)
        self.schedule = SimultaneousActivation(self)
        self.datacollector = DataCollector(model_reporters={"Grid" : getGrid})
        self.counter = 0

        self.floor = np.zeros((height, width))

        totalDirty = width * height * dirtPercentage
        for i in range(int(totalDirty)):
            finished = False
            while not finished:
                ren = int(np.random.rand() * 1000) % height
                col = int(np.random.rand() * 1000) % width
                if self.floor[ren][col] == 0:
                    self.floor[ren][col] = 1
                    finished = True

        for i in range(numAgents):
            a = Roomba(i, self)
            self.grid.place_agent(a, (1,1))
            self.schedule.add(a)

    def isAllClean(self):
        return np.all(self.floor == 0)
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.counter += 1

# --------------------------------------------- #
# ------------- INITIAL VARIABLES ------------- #
# --------------------------------------------- #

WIDTH = 25
HEIGHT = 15
NUM_AGENTS = 24
DIRT_PERCENTAGE = 0.4
MAX_ITER = 150

# --------------------------------------------- #
# --------- CREATION AND RUN OF MODEL --------- #
# --------------------------------------------- #

model = House(WIDTH, HEIGHT, NUM_AGENTS, DIRT_PERCENTAGE)
i = 1
finished = False
start_time = time.time()
while i <= MAX_ITER:
    if not model.isAllClean():
        model.step()
        i += 1
    else:
        finished = True
        break

totalMovements = 0
totalCellsCleaned = 0
for agent in model.schedule.agent_buffer():
    totalMovements += agent.movements
    totalCellsCleaned += agent.spacesCleaned

totalCells = WIDTH * HEIGHT
startDirty = totalCells * DIRT_PERCENTAGE
totalDirty = startDirty - totalCellsCleaned
totalClean = totalCells - totalDirty
percetageClean = totalClean / totalCells

print("EXECUTION TIME: ", str(datetime.timedelta(seconds=(time.time() - start_time))))

if finished:
    print("\nCLEAN HOUSE!")
else:
    print("\nCOULD NOT CLEAN THE WHOLE HOUSE")

print("STEPS EXECUTED: \t\t", model.counter)
print("TOTAL MOVEMENTS BY AGENTS: \t", totalMovements)
print("SPACES TO CLEAN: \t\t", int(startDirty))
print("TOTAL SPACES CLEANED: \t\t", totalCellsCleaned)
print("REMAINING SPACES TO CLEAN: \t", int(totalDirty))
out = percetageClean * 100
print("PERCENTAGE OF CLEAN SPACE: \t", "{:.2f}".format(out), "%")

# --------------------------------------------- #
# ------ GRAPHIC REPRESENTATION OF MODEL ------ #
# --------------------------------------------- #

gridColors = colors.ListedColormap(['#FFFFFF', '#6D543E', '#D5B090', '#BF9A78', '#A07E60', '#6D543E', '#919191'])
bounds = [0, 1, 2, 3, 4, 5, 6, 7]
norm = colors.BoundaryNorm(bounds, gridColors.N)

allGrid = model.datacollector.get_model_vars_dataframe()
fig, axs = plt.subplots(figsize = (4, 4))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(allGrid.iloc[0][0], cmap = gridColors, norm=norm)

def animate(i):
    patch.set_data(allGrid.iloc[i][0])

anim = animation.FuncAnimation(fig, animate, frames=model.counter)
plt.show()