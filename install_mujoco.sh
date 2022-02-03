#!/usr/bin/env bash
#
# This script installs Mujoco 2.1.1.
# TODO(kevin): Generalize to versions >= 2.1.1

wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
tar -xf mujoco-2.1.1-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
mv mujoco-2.1.1 ~/.mujoco
rm -r mujoco-2.1.1-linux-x86_64.tar.gz

# Install GLFW and GLEW for hardware accelerated rendering.
sudo apt install libglfw3 libglew2.1
