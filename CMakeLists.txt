cmake_minimum_required(VERSION 3.16.3)
project(gravity_compensation)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
# find_package(rclpy REQUIRED)

install(
  PROGRAMS
    src/traj_generation.py
    src/mujoco_simulation.py
    src/para_regression.py
    src/dynamic_model.py
    src/CollisionCheck.py
    src/admittance_test.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY src/resources urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
