# CSCI5552-Project-Spring2021

## About
This is the final project for CSCI5552-Sensing & Estimating at the University of Minnesota Twin-Cities
Authors: Bharath Sivaram & Isaac Kasahara

## Objective
Given a virtual environment, the robot must go from a starting location inside one corridor and
navigate to a specified goal inside the opposite corridor.

## Limitations
1. Robot can use noisy odometer, 1D LiDAR and a camera
2. Can use distinct markers (red spheres) to navigate in open area but use LIDAR to navigate corridors and avoid obstacles
3. Can use landmarks’ ids to identify them, but not allowed to use their global positions

## How to run
You must have [WeBots](https://www.cyberbotics.com/) installed to run.
Go into /worlds and open 'project.wbt'. The simulation should automatically start.
The goal position can be changed by editing line 225 of the 'mycontroller.py' file, which can be found in /controllers/my_controller.
The details/explanation of the chosen path planning can be found in the 'CSCI5552_Project_Report.pdf' file.
