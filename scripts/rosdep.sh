#!/bin/sh
# External, ROS and system package dependencies

PACKAGES="ros-groovy-openni-camera
          ros-groovy-openni-tracker
          ros-groovy-openni-launch"

sudo apt-get install $PACKAGES
