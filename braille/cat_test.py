# Extended demo of of the PCA9685 PWM servo/LED controller library.
# This will spell "CAT" in Braille using one character of solenoids.
# Author: Christopher Smith
from __future__ import division
import time

# Import the PCA9685 module.
import Adafruit_PCA9685

import pybrl

print pybrl.braille("cat")

# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

# Set frequency to max 1000Hz
pwm.set_pwm_freq(1000)

MOVE_PWM = 4095
ON_PWM = 2048
OFF_PWM = 0
MOVE_TIME = 0.05
ON_TIME = 3
OFF_TIME = 0.2

NUM_ROWS = 3
NUM_COLS = 2

while True:
    for letter in "cat":
        braille_matrix = pybrl.convert(letter, pybrl.matrixcodes, pybrl.asciicodes)[0]
        for row in xrange(NUM_ROWS):
            for col in xrange(NUM_COLS):
                if braille_matrix[row][col]:
                    dot_num = NUM_COLS * row + col
                    pwm.set_pwm(dot_num, 0, MOVE_PWM)
                    time.sleep(MOVE_TIME)
                    pwm.set_pwm(dot_num, 0, ON_PWM)
        time.sleep(ON_TIME)
        for i in xrange(NUM_ROWS*NUM_COLS):
            pwm.set_pwm(i, 0, OFF_PWM)
        time.sleep(OFF_TIME)
