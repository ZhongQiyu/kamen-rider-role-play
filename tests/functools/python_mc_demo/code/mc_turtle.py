# Part 2: MC Turtle

# library imports
from mcpi.minecraft import Minecraft
from mcpi import block
from minecraftstuff import MinecraftTurtle
from constants import *


def mc_turtle():
    """
    Demonstrate the way of using MinecraftTurtle with a
    quick example of drawing a number eight and a nine.
    """
    # connect to server and find player position
    print("To establish a Minecraft server connection.")
    mc = Minecraft.create("localhost", 4711)
    print("A Minecraft server connection is established.\n")
    my_id = mc.getPlayerEntityId(USERNAME)
    my_pos = mc.entity.getPos(my_id)

    # create an instance of the Turtle object
    print("To create a Minecraft Turtle object.")
    turtle = MinecraftTurtle(mc, my_pos)
    print("A Minecraft Turtle object is created.\n")

    # set the position of the turtle to be 1
    # unit away from the player in diagonal
    print("To add offset on the Turtle's position.")
    x = my_pos.x
    z = my_pos.z
    turtle.setx(x + 1)
    turtle.setz(z + 1)
    # remember to update the x and z to use later on
    x = x + 1
    z = z + 1
    print("Offset added.\n")

    # draw a number eight's bottom half with black wool
    print("To draw the bottom half of a number eight in black.")
    for i in range(SQUARE_EDGES):
        turtle.forward(NUM_WIDTH)
        turtle.right(TURN_DEG)
    print("Bottom half drawn.\n")

    # move to the place to start drawing the top half
    print("Halt painting.")
    turtle.penup()
    print("Move to the place to draw the top half.")
    turtle.right(TURN_DEG)
    turtle.forward(NUM_WIDTH)
    print("Continue painting.\n")
    turtle.pendown()

    # draw the top half
    print("To draw the top half of the number eight in black.")
    for i in range(SQUARE_EDGES):
        if i == SQUARE_EDGES - 1:
            # avoid repetitive drawing
            turtle.penup()
            turtle.forward(NUM_WIDTH)
            turtle.pendown()
        else:
            turtle.forward(NUM_WIDTH)
        turtle.left(TURN_DEG)
    print("Top half drawn.\n")

    # change the pen block to iron block
    turtle.penblock(block.IRON_BLOCK.id)
    print("Changed the pen block to iron block.\n")

    # mark out a number nine's top square
    print("To mark out the top square of a number nine.")
    for i in range(SQUARE_EDGES):
        turtle.forward(NUM_WIDTH)
        turtle.left(TURN_DEG)
    print("Top square marked out.\n")

    # turn from forward to backward
    turtle.right(TURN_DEG*2)
    print("Made a U-turn.\n")

    # draw the two strokes of number nine
    print("To draw the two strokes of the number nine.")
    for i in range(NINE_BOT_STROKES):
        turtle.forward(NUM_WIDTH)
        turtle.right(TURN_DEG)
    print("Strokes drawn.\n")

    # replace the left traces of the previous eight with air
    print("To replace the residuals of number eight with air.")
    # remember to get the y-coordinate
    y = my_pos.y
    for i in range(NUM_WIDTH - 1):
        mc.setBlock(x + NUM_WIDTH, y - 1, z + 1 + i, block.AIR)
    print("Residuals replaced.\n")

    # indicates that the second part ends
    print("Part 2 ends.")


if __name__ == "__main__":
    mc_turtle()
