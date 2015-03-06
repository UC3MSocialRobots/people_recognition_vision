#!/bin/sh
# External, ROS and system package dependencies

PACKAGES="ros-hydro-openni-camera
          ros-hydro-openni-tracker
          ros-hydro-openni-launch"

sudo apt-get install $PACKAGES
