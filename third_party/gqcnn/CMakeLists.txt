cmake_minimum_required(VERSION 3.8)
project(gqcnn)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GQCNNGrasp.msg"
  "msg/BoundingBox.msg"
  "msg/Action.msg"
  "msg/Observation.msg"
  "srv/GQCNNGraspPlanner.srv"
  "srv/GQCNNGraspPlannerBoundingBox.srv"
  "srv/GQCNNGraspPlannerSegmask.srv"
  DEPENDENCIES geometry_msgs sensor_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    scripts/grasp_planner.py
    scripts/grasp_planner_bb.py
    scripts/grasp_planner_segmask.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

ament_export_dependencies(
  rosidl_default_runtime
  geometry_msgs
  sensor_msgs
)

ament_package()
