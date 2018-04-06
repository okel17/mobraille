---
layout: default
title: April 5, 2018
description: Fourth Blog Post
---

# April 5 Weekly Update

__Software Team (Tara & Omkar)__
  - Omkar got segmentation fully working by using hsv and thresholding. They were able to successfully segment all 90 training images.
  - Tara got the full PCA and LDA implementations working.
  - She was able to transform the dataset from 90x1000000 to 90x90 using PCA and was able to transform the dataset to 90x2 using LDA.
  - Tara and Omkar got the SVM working and were able to classify a given test image taken inside the box by the Raspberry Pi.

__Hardware Team (Chris & Omkar)__
  - Check out this video of our solenoids the word "cat" in Braille: https://photos.app.goo.gl/psU9iChzsfYdF1nL2
  - For our mid-point demo, we turned 6 solenoids (1 Braille character) on and off simultaneously using Arduino GPIO pins and switching circuits.
  - Later in the week, we controlled each of the solenoids individually and used PWM to save power. The above video demonstrates this.
  - We're abandoning the PCB! It will barely arrive in time to be useful and the hours of assembly will eat up time that we could otherwise spend writing software and integrating parts. This change of plans was approved by our TA.
  
This coming week we will:
  - Chris will work on controlling the solenoids from the Raspberry Pi.
  - Chris will create a Python API that translates a given string into Braille for display by the solenoids.
  - Tara and Omkar will work on improving the accuracy of the ML model by experimenting with which and how many eigenvectors they keep in PCA and LDA.
  - Omkar will work on just using 0s and 1s to represent the shape of the toy in the image to reduce the amount of data.
  - Tara will work on speeding up the ML pipeline.

[Back](../index.md)
