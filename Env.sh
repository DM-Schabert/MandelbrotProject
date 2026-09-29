# Shared environment for the Mandelbrot project on DTU HPC.
# Sourced by both the Makefile and batch.sh, so the module versions
# only need to be changed here.

# Make the `module` command available in non-interactive shells
if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi

module load python3/3.13.11
module load mpi/5.0.3-gcc-13.3.0-binutils-2.42