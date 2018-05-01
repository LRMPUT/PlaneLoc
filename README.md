# PlaneLoc

An open source project that provides a probabilistic framework
for global localization using segmented planes.

Prerequesties:  
-Boost  
-Eigen  
-PCL 1.8  
-OpenCV >= 3.0  
-g2o

### Building:  

Tested on Ubuntu 14.04.  
1. Install Boost and Eigen:
```commandline
sudo apt-get install libboost-system-dev libboost-filesystem-dev libeigen3-dev
```
Please ensure that your Eigen version is not beta version. Version 3.3 beta contains a bug that causes system to produce wrong results.

2. Build PCL from sources and install it:
```commandline
sudo apt-get install libvtk6-dev libflann-dev libxi-dev libxmu-dev libgtest-dev libproj-dev
wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.0.tar.gz
tar xvfj pcl-pcl-1.8.0.tar.gz && cd pcl-pcl-1.8.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```
3. Build OpenCV from sources and install it:
```commandline
wget https://sourceforge.net/projects/opencvlibrary/files/opencv-unix/3.1.0/opencv-3.1.0.zip
unzip opencv-3.1.0.zip && cd opencv-3.1.0
mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install
```
4. Build g2o and install it:
```commandline
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```
5. Build PlaneLoc:
```commandline
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Launching:  

1. Adjust settings file in _res/settings.yml_ for your dataset.  
2. Launch demo:

```commandline
./demoPlaneSlam -s ../res/settings.yml
```