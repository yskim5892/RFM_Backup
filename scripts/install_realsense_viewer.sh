sudo apt update
sudo apt install -y \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    curl
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
  | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" \
| sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt update
sudo apt install -y \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-dkms
