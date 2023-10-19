git clone https://gcc.gnu.org/git/gcc.git gcc-source
cd gcc-source/ || exit
git checkout remotes/origin/releases/gcc-12

sudo apt-get install gcc-multilib flex libmpfrc++-dev libmpc-dev libgmp-dev
./contrib/download_prerequisites

mkdir ../gcc-12-build
cd ../gcc-12-build/ || exit
./../gcc-source/configure --prefix="$HOME"/install/gcc-12 --enable-languages=c,c++

make -j16
make install

sudo update-alternatives --install /usr/bin/gcc gcc ~/install/gcc-12/bin/gcc 100 --slave /usr/bin/g++ g++ ~/install/gcc-12/bin/g++ --slave /usr/bin/gcov gcov ~/install/gcc-12/bin/gcov
sudo update-alternatives --install /usr/bin/gcc-12 gcc-12 ~/install/gcc-12/bin/gcc 100
