import RPi.GPIO as GPIO
import time
import os

BUTTON_PIN = 17

GPIO.setmode(GPIO.BCM)

GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while True:
    input_state = GPIO.input(BUTTON_PIN)
    if input_state == False:
        print('Button Pressed')
        os.system('python ./braille/n_char.py "go"')
        os.system('./run_demo.sh')
        time.sleep(0.2)
