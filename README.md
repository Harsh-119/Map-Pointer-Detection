# Map Pointer Detection

<div align="center">
<img src="map_coordinate_system.png">
</div>

## Overview

This project involves developing a system to facilitate a Virtual Reality (VR) reconstruction of a region of London, showing how it appeared around 1900. The system enables visitors to interact with a historical map of London (Horwood’s hand-drawn map from around 1792) to determine locations and directions in the VR reconstruction.

Visitors will use a red pointer placed on a printed map to interact with the VR system. The map is viewed from above with a camera, and the goal is to segment the map and pointer from the image, and then determine the location and orientation of the pointer to update the VR view.


The system performs the following tasks:
1. **Segment the Map**: Extract the map from the blue background, ensuring the map edges align with the edges of the extracted image.
2. **Segment the Red Pointer**: Identify and extract the red pointer from the image.
3. **Locate the Tip of the Pointer**: Determine the location of the pointer tip to identify the viewpoint.
4. **Determine the Orientation of the Pointer**: Convert the orientation of the pointer to a bearing and output the results.

## Usage

The program will take the input as `python3 mapreader.py develop/develop-001.png` and it will output the coordinates , position and bearing, from the map. There are eight pictures, in which the red arrow is pointing to different locations to showcase the program.


- This command will run the program on all images in one go.

######
    for file in *.jpg; do python3 mapreader.py "$file" done

## Sample Output
The filename to work on is develop-001.jpg.
POSITION 0.442 0.598
BEARING 262.9
