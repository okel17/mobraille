---
layout: default
title: April 12, 2018
description: Fifth Blog Post
---

# April 12 Weekly Update

__Software Team (Tara & Omkar)__
  - Omkar began work on setting up Raspberry Pi to work headless on CMU network and began exploring how to integrate the pipeline with Amazon WebServices to host the bulk of the computation.
  - Omkar added a binary threshold to the segmentation code to lessen the computation complexity for PCA and LDA.
  - Tara worked on improving the classifier by changing the eigenvectors kept after both PCA and LDA. She also has been working on quantifying the performance of the model and integrating with Omkar’s new segmentation code in addition to speeding up classification.

__Hardware Team (Chris & Omkar)__
  - Chris controlled some of the solenoids from the Raspberry Pi with the Python RPi GPIO library.
  - Chris acquired a power supply rated for 35A at 5V for that should be appropriate for the team's needs.
  
This coming week we will:
  - Chris will attempt to power the solenoids with the power supply.
  - Chris will continue working on controlling the solenoids from the Raspberry Pi with Python.
  - Tara and Omkar will work on improving the accuracy of the ML model.
  - Omkar will continue working on networking the Raspberry Pi and connecting it to an AWS server.
  
[Back](../index.md)
