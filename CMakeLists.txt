### 1 - Required CMake Version
cmake_minimum_required(VERSION 2.8.3)
### 2- Package Name
project(people_recognition_vision)
### 3 - Find other CMake/Catkin packages needed for build
SET(ROSDEPS message_generation  people_msgs)
find_package(catkin REQUIRED COMPONENTS ${ROSDEPS})
### 4 - Message/Service/Action Generators (add_message_files(), add_service_files(), add_action_files())
add_service_files(
  FILES
  MatchPPL.srv
)
### 5 - Invoke message/service/action generation (generate_messages())
generate_messages(
  DEPENDENCIES
  people_msgs # Or other packages containing msgs
)
### 6 - Specify package build info export (catkin_package())
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS ${ROSDEPS}
  DEPENDS
)
### 7 - Libraries/Executables to build (add_library()/add_executable()/target_link_libraries())
include_directories(
  include ${catkin_INCLUDE_DIRS}
  ${people_msgs_INCLUDE_DIRS}
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
