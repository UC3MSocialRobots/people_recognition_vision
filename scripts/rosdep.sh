#!/bin/sh
# External, ROS and system package dependencies

PACKAGES="ros-`rosversion -d`-openni-camera
          ros-`rosversion -d`-openni-tracker
          ros-`rosversion -d`-openni-launch"

sudo apt-get install $PACKAGES
