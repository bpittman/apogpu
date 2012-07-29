cmake_minimum_required (VERSION 2.6)
project (apogpu)

include_directories(/home/uahcls14/local/include )
link_directories( /home/uahcls14/local/lib )

add_executable(sfprocess sfprocess.c)
add_executable(sfdistortion sfdistortion.c)
add_executable(sfdelay sfdelay.c)
add_executable(sfcompare sfcompare.c)
add_executable(sfloop sfloop.c)
add_executable(sflowpass sflowpass.c)

target_link_libraries(sfprocess sndfile)
target_link_libraries(sfdistortion sndfile)
target_link_libraries(sfdelay sndfile)
target_link_libraries(sfcompare sndfile)
target_link_libraries(sfloop sndfile)
target_link_libraries(sflowpass sndfile)
