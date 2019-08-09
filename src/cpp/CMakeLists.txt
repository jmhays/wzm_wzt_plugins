# Defines targets for the C++ restraints implemented here. These CMake targets are used by the
# unit tests and by the Python module target defined in ../pythonmodule/CMakeLists.txt

# Create a shared object library for a simple restraint.
add_library(production_potential STATIC
            production_potential.h
            production_potential.cpp
            sessionresources.cpp)
set_target_properties(production_potential PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(production_potential PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:include>
                           )
# RPATH management
# don't skip the full RPATH for the build tree
set_target_properties(production_potential PROPERTIES SKIP_BUILD_RPATH FALSE)

# (but later on when installing)
# If building with setuptools, CMake will not be performing the install
set_target_properties(production_potential PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

target_link_libraries(production_potential PRIVATE Gromacs::gmxapi)

# Create a shared object library for a simple restraint.
add_library(convergence_potential STATIC
            convergence_potential.h
            convergence_potential.cpp
            sessionresources.cpp)
set_target_properties(convergence_potential PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(convergence_potential PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:include>
                           )
# RPATH management
# don't skip the full RPATH for the build tree
set_target_properties(convergence_potential PROPERTIES SKIP_BUILD_RPATH FALSE)

# (but later on when installing)
# If building with setuptools, CMake will not be performing the install
set_target_properties(convergence_potential PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

target_link_libraries(convergence_potential PRIVATE Gromacs::gmxapi)

# Create a shared object library for a simple restraint.
add_library(training_potential STATIC
            training_potential.h
            training_potential.cpp
            sessionresources.cpp)
set_target_properties(training_potential PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(training_potential PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:include>
                           )
# RPATH management
# don't skip the full RPATH for the build tree
set_target_properties(training_potential PROPERTIES SKIP_BUILD_RPATH FALSE)

# (but later on when installing)
# If building with setuptools, CMake will not be performing the install
set_target_properties(training_potential PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

target_link_libraries(training_potential PRIVATE Gromacs::gmxapi)

# Create a shared object library for a simple restraint.
add_library(ensemble_potential STATIC
            ensemblepotential.h
            ensemblepotential.cpp
            sessionresources.cpp)
set_target_properties(ensemble_potential PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ensemble_potential PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:include>
                           )
# RPATH management
# don't skip the full RPATH for the build tree
set_target_properties(ensemble_potential PROPERTIES SKIP_BUILD_RPATH FALSE)

# (but later on when installing)
# If building with setuptools, CMake will not be performing the install
set_target_properties(ensemble_potential PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

target_link_libraries(ensemble_potential PRIVATE Gromacs::gmxapi)