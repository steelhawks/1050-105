"""
OpenCV RGB color schemes
 ( RED , GREEN , BLUE )
"""
from random import randint

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def random():
    return (randint(0, 255),
            randint(0, 255),
            randint(0, 255))
