# Remove flags setup from cxx-compiler
unset CFLAGS
unset CPPFLAGS
unset CXXFLAGS
unset DEBUG_CFLAGS
unset DEBUG_CPPFLAGS
unset DEBUG_CXXFLAGS
unset LDFLAGS

# Setup ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Create compile_commands.json for language server
export CMAKE_EXPORT_COMPILE_COMMANDS=1

# Activate color output with Ninja
export CMAKE_COLOR_DIAGNOSTICS=1

# Set default build value only if not previously set
export EIGENPY_BUILD_TYPE=${EIGENPY_BUILD_TYPE:=Release}
export EIGENPY_PYTHON_STUBS=${EIGENPY_PYTHON_STUBS:=ON}
export EIGENPY_CHOLMOD_SUPPORT=${EIGENPY_CHOLMOD_SUPPORT:=OFF}
export EIGENPY_ACCELERATE_SUPPORT=${EIGENPY_ACCELERATE_SUPPORT:=OFF}
