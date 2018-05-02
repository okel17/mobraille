#!/usr/bin/env python
# Usage: ./one_char.py "string to print"
# This will spell the given string in Braille with one character of solenoids.
# Author: Christopher Smith, Omkar Kelkar
from __future__ import division
import Adafruit_PCA9685
import pybrl
import sys
import time
import os


# N number of characters to display #
N = 2

# freq in Hz #
PWM_FREQ = 1000 # maximum PWM frequency of the PCA9685

# pulse widths between 0 and 4095 #
MOVE_PWM = 4095     # width required to move solenoid to on position
ON_PWM = 2048       # width required to keep solenoid in on position
OFF_PWM = 0         # width that turns the solenoid off

# times in seconds #
MOVE_TIME = 0.08    # time that MOVE_PWM is on
ON_TIME = 1         # time that each character is displayed
OFF_TIME = 0.2      # time between each character when all solenoids are off

# Braille character dimensions #
NUM_ROWS = 3
NUM_COLS = 2
# the dots of a single character are numbered as follows:
#   0 1    6 7    12 13   ...
#   2 3    8 9    14 15   ...
#   4 5   10 11   16 17   ...

if __name__ == "__main__":
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(PWM_FREQ)

    user_string = sys.argv[1]
    remainder = N - (len(user_string) % N)
    padding = " "*remainder
    user_string = user_string + padding

    os.system("pico2wave -w ./out.wav " + user_string + " && aplay ./out.wav")

    print "Displaying ", user_string, "with the following Braille:"
    print pybrl.braille(user_string)


    groupedLetters = [user_string[i:i+N] for i in range(0, len(user_string), N)]

    groupedBrailleMatrix = []
    for group in groupedLetters:
        braille_matrix = pybrl.convert(group, pybrl.matrixcodes, pybrl.asciicodes)
        groupedBrailleMatrix.append(braille_matrix)

    for group in groupedBrailleMatrix:
        for characterNumber in xrange(len(group)):
	    for row in xrange(NUM_ROWS):
	        for col in xrange(NUM_COLS):
                    if group[characterNumber][row][col]:
                        dot_num = NUM_COLS * row + col + 6*characterNumber
                        pwm.set_pwm(dot_num, 0, MOVE_PWM)
                        time.sleep(MOVE_TIME)
                        pwm.set_pwm(dot_num, 0, ON_PWM)
        time.sleep(ON_TIME)
        for i in xrange(NUM_ROWS*NUM_COLS*len(group)):
            pwm.set_pwm(i, 0, OFF_PWM)
        time.sleep(OFF_TIME)
