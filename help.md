# GPU 

## CUDA Toolkit

The NVIDIA® CUDA® Toolkit provides a development environment for creating high performance GPU-accelerated applications. With the CUDA Toolkit, you can develop, optimize, and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler, and a runtime library to build and deploy your application on major architectures including x86, Arm and POWER.

you can check for systems GPU support in chrome tab 
chrome://gpu


Add NVIDIA package repositories
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
```

Install NVIDIA driver
```
sudo apt-get install --no-install-recommends nvidia-driver-450
```
Reboot. Check that GPUs are visible using the command: nvidia-smi

Install development and runtime libraries (~4GB)
```
sudo apt-get install --no-install-recommends \
    cuda-10-1 \
    libcudnn7=7.6.5.32-1+cuda10.1  \
    libcudnn7-dev=7.6.5.32-1+cuda10.1
```

Install TensorRT. Requires that libcudnn7 is installed above.
```
sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1
```


## Issues

**Issue1** lua.h not found 
```
lnn.c:1:9: fatal error: lua.h: No such file or directory
 #include<lua.h>
```
\
**solution** on ubuntu get liblua installed 
```
sudo apt-get install liblua5.3-0 liblua5.3-dev
```

**Issue2** luarocks install gives no matching results 
```
Error: No results matching query were found.
```
\
**solution** I was stick on it too ans realised after a while that the path of luarocks "which luarocks" 
should show torch "~/torch/install/bin/luarocks"
and not the genric "/usr/local/bin/luarocks"

ref : https://groups.google.com/g/torch7/c/LLw2rHYpapg

So I followed the installtion process for torch  
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; 
bash install-deps;
./install.sh
``` 
ref : http://torch.ch/docs/getting-started.html

to add torch to PATH 
```
source ~/.bashrc
```
or add full path before installation such as 
```
~/torch/install/bin/luarocks install torch 
```

**Issue3** GPU Training - Installing CUDA support for GPU  
```
CMake Error at /usr/share/cmake-3.10/Modules/FindCUDA.cmake:682 (message):
  Specify CUDA_TOOLKIT_ROOT_DIR
Call Stack (most recent call first):
  CMakeLists.txt:7 (FIND_PACKAGE)
```
\
**solution** 
Torch allows to easily train models with GPU acceleration through packages such as cutorch and cunn
For ubuntu 18 refer to nvidia site
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```
ref : https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork

I used 
```
lab_release -a

Distributor ID: Ubuntu
Description:    Ubuntu 18.04.4 LTS
Release:        18.04
Codename:       bionic
```

with configuration
```
lscpu 

Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              4
On-line CPU(s) list: 0-3
Thread(s) per core:  2
Core(s) per socket:  2
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               142
Model name:          Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz
Stepping:            9
CPU MHz:             799.810
CPU max MHz:         3500.0000
CPU min MHz:         400.0000
BogoMIPS:            5808.00
Virtualization:      VT-x
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            4096K
NUMA node0 CPU(s):   0-3
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp md_clear flush_l1d
```


**Issue4** lua <script> fails 
```
lua: error loading module 'libpaths' from file '/home/altanai/torch/install/lib/lua/5.1/libpaths.so':
        /home/altanai/torch/install/lib/lua/5.1/libpaths.so: undefined symbol: luaL_register
```
/
**solution** preload lua
```
LD_PRELOAD=~/torch/install/lib/libluajit.so th
> train/lua
```


**Issue5** FindCUDA.cmake is trying to find my CUDA installation directory and failing
```
CMake Error at /usr/share/cmake-3.10/Modules/FindCUDA.cmake:682 (message):
  Specify CUDA_TOOLKIT_ROOT_DIR
```
\
**solution** To deploy GPU-Accelerated Apps with CUDA toolkit , install
```
sudo apt-get -y install cuda
```

**Issue6** make for CUDA
```
CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
Please set them or make sure they are set and tested correctly in the CMake files:
CUDA_cublas_device_LIBRARY (ADVANCED)
    linked by target "THC" in directory /tmp/luarocks_cutorch-scm-1-856/cutorch/lib/THC
```
\
**solution**
update the cmake from version 3.10.x to latest preferabbly 3.14x or avbove 
remove teh existing default cmake version from ubuntu 
```
sudo apt remove cmake cmake-data 
```
install latest form cmake site 
```
https://cmake.org/download/
```
After installing the sh file check the version 
```
cmake-3.17.5-Linux-x86_64/bin/cmake -version
cmake version 3.17.5
```
