cmake_minimum_required(VERSION 3.8)
project(drone_mission)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  autonomous_drone_mission.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME autonomous_drone_mission
)

install(PROGRAMS
  lidar_mapper.py
  qr_stream.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  mission_launch.xml
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
