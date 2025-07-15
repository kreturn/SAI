cmake_minimum_required (VERSION 2.8)
set(PROJ_NAME "sainversion")
project (${PROJ_NAME})

set(CMAKE_C_COMPILER gcc ) #gcc ) #mpicc)
set(CMAKE_CXX_COMPILER g++ ) #g++ ) #mpic++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
#set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -fopenmp -lgomp")

set(Eigen3_DIR ${CMAKE_SOURCE_DIR}/lib/Eigen)
include_directories(${Eigen3_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/*.h")
file(GLOB EXESOURCES "${CMAKE_SOURCE_DIR}/src/*.cxx")

add_library(${PROJ_NAME} ${SOURCES} ${SOURCESP} ${HEADERS} ${Eigen3_DIR})
set_target_properties( ${PROJ_NAME}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

set(PROJEXE "sai")
set(SOURCE_FILES src/main.cc) 

add_executable(${PROJEXE} ${SOURCE_FILES} ${HEADERS} ${Eigen3_DIR}) 
target_link_libraries (${PROJEXE} ${EXTRA_LIBS} ${PROJ_NAME})
set_target_properties(${PROJEXE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

foreach(appsrc ${EXESOURCES})
    get_filename_component(appname ${appsrc} NAME_WE)
    add_executable(${appname} ${appsrc} )
    target_link_libraries(${appname} ${EXTRA_LIBS} ${PROJ_NAME})
    set_target_properties(${appname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
endforeach(appsrc ${EXESOURCES})
