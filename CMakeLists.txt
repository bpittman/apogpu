cmake_minimum_required (VERSION 2.6)
project (apogpu)
add_executable(sfprocess sfprocess.c)
add_executable(sfdistortion sfdistortion.c)
add_executable(sfdelay sfdelay.c)

target_link_libraries(sfprocess sndfile)
target_link_libraries(sfdistortion sndfile)
target_link_libraries(sfdelay sndfile)
