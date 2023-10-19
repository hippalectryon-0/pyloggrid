sudo apt-get install cmake build-essential
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.7/llvm-project-15.0.7.src.tar.xz
tar -xf llvm-project-15.0.7.src.tar.xz
cd llvm-project-15.0.7.src || exit
mkdir build
cd build || exit
# If necessary: export LD_LIBRARY_PATH=~/install/gcc-12/lib64  <- path to your gcc install
cmake -DLLVM_ENABLE_PROJECTS="clang;openmp" -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm
make -j16
cp projects/openmp/runtime/src/omp.h lib/clang/15.0.7/include # because otherwise the current install doesn't work with openmp ??!!
sudo cp ./lib/libomp.so /usr/lib
#make check-clang

sudo update-alternatives --install /usr/bin/clang-15 clang-15 ~/llvm-project-15.0.7.src/build/bin/clang 100 --slave /usr/bin/clang++-15 clang++-15 ~/llvm-project-15.0.7.src/build/bin/clang++
sudo update-alternatives --install /usr/bin/clang clang ~/llvm-project-15.0.7.src/build/bin/clang 100

# # If Ubuntu<20: Setup CMake because the system one is too old
#wget https://github.com/Kitware/CMake/releases/download/v3.23.4/cmake-3.23.4-linux-x86_64.tar.gz
#echo "3fbcbff85043d63a8a83c8bdf8bd5b1b2fd5768f922de7dc4443de7805a2670d  cmake-3.23.4-linux-x86_64.tar.gz" | sha256sum -c
#tar -xf cmake-3.23.4-linux-x86_64.tar.gz
## Done, cmake is usable, nothing is installed on the system, everything is self-contained *inside* the build directory itself.
#export PATH=$PWD/cmake-3.23.4-linux-x86_64/bin/:$PATH
