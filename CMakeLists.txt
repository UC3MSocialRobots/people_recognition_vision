### 1 - Required CMake Version
cmake_minimum_required(VERSION 2.8.3)
### 2- Package Name
project(people_recognition_vision)
### 3 - Find other CMake/Catkin packages needed for build
find_package(catkin REQUIRED COMPONENTS
  vision_utils people_msgs vision_utils kinect)
find_package(PCL REQUIRED)
### 4 - Message/Service/Action Generators (add_message_files(), add_service_files(), add_action_files())
# Nothing in this package
### 5 - Invoke message/service/action generation (generate_messages())
# Nothing in this package
### 6 - Specify package build info export (catkin_package())
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp
  DEPENDS vision_utils people_msgs vision_utils kinect
)
### 7 - Libraries/Executables to build (add_library()/add_executable()/target_link_libraries())
include_directories(
  include ${catkin_INCLUDE_DIRS}
  ${vision_utils_INCLUDE_DIRS}
  ${people_msgs_INCLUDE_DIRS}
  ${vision_utils_INCLUDE_DIRS}
  /usr/include/ni/
  ${PCL_INCLUDE_DIRS}
)
SET(AD_USE_ROS 1)
### 8 - Tests to build (catkin_add_gtest())
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
### 9 - Install rules (install())
install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h"
   PATTERN ".svn" EXCLUDE
)
