# Part 1: MCPI

# library imports
import random
from mcpi.minecraft import Minecraft
from mcpi import block
from constants import *


def minecraftpi():
    """
    Demostrate the way of using MCPI with a quick example
    of setting a Nether Portal randomly near to the player.
    """
    # create a MC server instance
    print("To establish a Minecraft server connection.")
    mc = Minecraft.create("localhost", 4711)
    print("A Minecraft server connection is established.\n")

    # find the player's coordinates
    print("To find the player's position.")
    my_id = mc.getPlayerEntityId(USERNAME)
    my_pos = mc.entity.getPos(my_id)
    x = my_pos.x
    y = my_pos.y
    z = my_pos.z
    print(f"The player's position is found as {(x, y, z)}.\n")

    # set the player's position into a new one
    # in MC, x means longitude, y means height, and z means latitude
    print("To reset the player's position.")
    x_off = random.randint(OFF_MIN, OFF_MAX)
    y_off = random.randint(OFF_MIN, OFF_MAX)
    z_off = random.randint(OFF_MIN, OFF_MAX)
    off_scale = 1
    new_x = x+off_scale*x_off
    new_y = y+off_scale*y_off
    new_z = z+off_scale*z_off
    mc.player.setPos(new_x, new_y, new_z)
    print(f"The player's position is reset from {(x, y, z)} to {(new_x, new_y, new_z)}.\n")

    # set up the Nether Portal
    # build the base and the top
    print("To build the base and the top of the Nether Portal.")
    for offset in range(NETHER_WIDTH):
        mc.setBlock(x + BASE + offset, y, z + BASE, block.OBSIDIAN)
        mc.setBlock(x + BASE + offset, y + NETHER_WIDTH, z + BASE, block.OBSIDIAN)
    print("The base and the top of the Portal is built.\n")
    # fill the left and the right
    print("To fill the left and the right of the Portal.")
    for offset in range(NETHER_HEIGHT - 1 - 1):
        mc.setBlock(x + BASE, y + BASE + offset, z + BASE, block.OBSIDIAN)
        mc.setBlock(x + NETHER_WIDTH, y + BASE + offset, z + BASE, block.OBSIDIAN)
    print("The left and the right of the Portal are filled.\n")

    # indicate that the first part ends
    print("Part 1 ends.")


if __name__ == "__main__":
    minecraftpi()
