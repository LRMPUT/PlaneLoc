# PlaneLoc

An open source project that provides a probabilistic framework
for global localization using segmented planes.

Prerequesties:  
-Boost  
-Eigen  
-PCL 1.8  
-OpenCV >= 3.0  
-g2o  
-CGAL

### Paper

If you find PlaneLoc useful in your academic work please cite the following paper:

    @article{wietrzykowski2019,
        title = {{PlaneLoc}: Probabilistic global localization in {3-D} using local planar features},
        author = {Jan Wietrzykowski and Piotr Skrzypczy\'{n}ski},
        journal = {Robotics and Autonomous Systems},
        volume = {113},
        pages = {160 - 173},
        year = {2019},
        issn = {0921-8890},
        doi = {https://doi.org/10.1016/j.robot.2019.01.008},
        url = {http://www.sciencedirect.com/science/article/pii/S0921889018303701},
        keywords = {Global localization, SLAM, Planar segments, RGB-D data},
    }


### Building:  

Tested on Ubuntu 16.04.  
1. Install Boost, Eigen, and CGAL:
```commandline
sudo apt-get install libboost-system-dev libboost-filesystem-dev libeigen3-dev libcgal-dev
```
2. Build PCL from sources and install it:
```commandline
sudo apt-get install libvtk6-dev libflann-dev libxi-dev libxmu-dev libgtest-dev libproj-dev
wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.1.tar.gz
tar xvfj pcl-pcl-1.8.1.tar.gz && cd pcl-pcl-1.8.1
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

### Dataset

Dataset is available [here](http://lrm.put.poznan.pl/rgbdw/).

### Building your own maps

Accumulated representation files can be generated by setting `processFrames` to 1 in settings file. Map is saved in file `../output/acc/acc%05d` every time new `accFrames` are processed, so for maps of the whole environment set `accFrames` to the number frames in the whole trajectory. For local maps set it to the desired length (usually 50 frames is a good compromise between the map size and the frequency of the localization). If you want to use precomputed local maps set `processFrames` to 0 and place generated maps in `acc` folder in a trajectory folder.

### Launching:  

1. Adjust settings file in _res/settings.yml_ for your dataset.  
2. Launch demo:

```commandline
./demoPlaneSlam -s ../res/settings.yml
```
