#This script will ping all of the solenoids in the solenoid array


#!/usr/bin/env python
# Usage: ./ping_all.py

# Author: Omkar Kelkar
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
    pwm0 = Adafruit_PCA9685.PCA9685(address=0x40)
    pwm1 = Adafruit_PCA9685.PCA9685(address=0x41)

    pwm0.set_pwm_freq(PWM_FREQ)
    pwm1.set_pwm_freq(PWM_FREQ)

    for i in range(0, 16):
        pwm0.set_pwm(i, 0, MOVE_PWM)
        time.sleep(MOVE_TIME)
        pwm0.set_pwm(i, 0, ON_PWM)
        time.sleep(ON_TIME)
        pwm0.set_pwm(i, 0, OFF_PWM)
        time.sleep(OFF_TIME)

    for i in range(0, 14):
        pwm1.set_pwm(i, 0, MOVE_PWM)
        time.sleep(MOVE_TIME)
        pwm1.set_pwm(i, 0, ON_PWM)
        time.sleep(ON_TIME)
        pwm1.set_pwm(i, 0, OFF_PWM)
        time.sleep(OFF_TIME)


