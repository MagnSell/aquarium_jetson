# Digital twin of an aquarium
This repository contains all the code necessary to create a digital twin of an aquarium on a Jetson Orin Nano developer kit.

### Installation
- Download pytorch and torchvision following this guide: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048.
    - With Jetpack 5.1 : PyTorch v2.1 - torchvision v0.16.1
 
- Downlad OpenCV for CUDA following this guide: https://github.com/Qengineering/Install-OpenCV-Jetson-Nano.
    - Version 4.8.0
- Download the other libraries in requirements.txt
- Create a .env file containing DATABASE_URL to the database you wish to upload.
- Create a logs folder for sensor measurement logs.

### Running
- Navigate to the repository in the terminal before running.
#### Main
- To run standard: sudo -E python3 main.py
##### Flags
- --viewer: toggles wether or not to display the videofeed locally. Default False.
- --logger: toggles wether or not to save the sensor measurements to a log file. Default False.
#### Variations
- To run purely sensor measurements: sudo -E python3 sensor_loop.py
- To run purely custom object detection: sudo -E python3 detector.py
