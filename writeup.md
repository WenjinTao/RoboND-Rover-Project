# Project: Search and Sample Return
---

## 1. Introduction

This project is to learn and implement the three essential elements of robotic, which are **perception**, **decision making** and **actuation**. The project environment is built with the Unity game engine.

![alt text][image1]

The main project steps are listed as follows:

**Training / Calibration Steps**

* The simulator is used to record data in "Training Mode"
* The [Jupyter Notebook](./code/Rover_Project_Test_Notebook.ipynb) is used to test the functions
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` created in this step should demonstrate that the mapping pipeline works.
* Use `moviepy` to process the images in the saved dataset with the `process_image()` function.  The produced video is also included.

**Autonomous Navigation / Mapping Steps**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what was done with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on the perception and decision functions until the rover does a reasonable (need to define metric) job of navigating and mapping. 

[//]: # "Image References"

[image1]: ./misc/rover_image.jpg
[worldmap]: ./output/worldmap.png
[rock_img]: ./calibration_images/example_rock1.jpg
[stuck_issue]: ./output/stuck_issue.PNG
[segmentation]: ./output/segmentation.png

## 2. Notebook Analysis
### 2.1 Color-based Segmentation

The color-based segmentation is realized by the function of `color_thresh()`, the code snippet is:

```python
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_navigable = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_navigable[above_thresh] = 1
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     color_navigable = cv2.morphologyEx(color_navigable, cv2.MORPH_CLOSE, kernel)
    
    # Detecing rock in HSV space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    color_rock = cv2.inRange(img_hsv, np.array([20, 100, 100]), np.array([30, 255, 255])) / 255
    
    # The left pixels are taken as obstacles
    color_obstacle = 1 - color_navigable - color_rock
    # There will have some mismatch between Navigable and Rock after the above mophological transformation, so set the negtive values to zero
    color_obstacle[color_obstacle<0] = 0
       
    # Return the binary image
    return color_navigable, color_obstacle, color_rock
```

where the navigable terrain is selected in the RGB space, while the golden rock is selected in the HSV space. Then the left pixels are taken as obstacles in the map. The following figures show a segmentation example.

![alt text][rock_img]

![][segmentation]

### 2.2 Image Processing Pipeline

The whole image processing pipeline is implemented in the function of `process_image()`, which follows the steps below:

- Perspective transformation: transform the robot view to bird-eye view
- Color segmentation: use color threshold in different color space to segment navigable terrain, rock and obstacle.
- Coordinate transformation: 
  - from image pixel to rover-centric coordinate
  - from rover-centric coordinate to world coordinate
- Maps updating: update the three maps (navigable, rock and obstacle).
- Maps visualization

```python
def process_image(img):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values 
    # print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])
    
    # 1) Define source and destination points for perspective transform
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    dst_size = 5 
    bottom_offset = 6
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                      [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                      ])
    # 2) Apply perspective transform
    img_warped = perspect_transform(img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    #    which returns a binary threshed image
    navigable_threshed, obstacle_threshed, rock_threshed = color_thresh(img_warped)
    # 4) Convert thresholded image pixel values to rover-centric coords
    navigable_x_pix, navigable_y_pix = rover_coords(navigable_threshed)
    obstacle_x_pix, obstacle_y_pix = rover_coords(obstacle_threshed)
    rock_x_pix, rock_y_pix = rover_coords(rock_threshed)
    # 5) Convert rover-centric pixel values to world coords
    scale = 10
    navigable_x_world, navigable_y_world = pix_to_world(navigable_x_pix, navigable_y_pix,
                                                        data.xpos[data.count],
                                                        data.ypos[data.count], 
                                                        data.yaw[data.count],
                                                        data.worldmap.shape[0], scale)
    obstacle_x_world, obstacle_y_world = pix_to_world(obstacle_x_pix, obstacle_y_pix,
                                                        data.xpos[data.count],
                                                        data.ypos[data.count], 
                                                        data.yaw[data.count],
                                                        data.worldmap.shape[0], scale)    
    rock_x_world, rock_y_world = pix_to_world(rock_x_pix, rock_y_pix,
                                                        data.xpos[data.count],
                                                        data.ypos[data.count], 
                                                        data.yaw[data.count],
                                                        data.worldmap.shape[0], scale)
    # 6) Update worldmap (to be displayed on right side of screen)
        # Example: data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          data.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          data.worldmap[navigable_y_world, navigable_x_world, 2] += 1        
    data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    data.worldmap[rock_y_world, rock_x_world, 1] += 1
    data.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 7) Make a mosaic image, below is some example code
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    # image(221)
    output_image[0:img.shape[0], 0:img.shape[1]] = img
        # Let's create more images to add to the mosaic, first a warped image
        # Add the warped image in the upper right hand corner
    # image(222)
    output_image[0:img.shape[0], img.shape[1]:] = img_warped    
    # image(223)
    output_image[img.shape[0]:img.shape[0]*2, 0:img.shape[1]] = np.dstack((navigable_threshed*255,                                                                           navigable_threshed*255,                                                                          navigable_threshed*255)).astype(np.float)

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.2, 0)        
        # Flip map overlay so y-axis points upward and add to output_image 
    # image(224)
    output_image[img.shape[0]:, image.shape[1]:image.shape[1]+data.worldmap.shape[1]] = np.flipud(map_add)
        # Then putting some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image
```

The above function returns a worldmap of a timestamp. As shown in the following figure, the segmentation is represented by different color channels: blue is navigable terrain, green is rock sample and red is obstacle region.

![alt text][worldmap]

Then `process_image()` was implemented on the test data using the `moviepy` functions. it created a video output of the result, which can be found [here](./output/test_mapping.mp4). 

### 2.3 Autonomous Navigation and Mapping

For the `perception_step() ` in ` perception.py` script, to improve the worldmap fidelity, the update function is modified as follows. The maps only get updated when the robot is in a relatively stable state.

```python
if Rover.roll < 1 or Rover.pitch < 1:
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
```

![][stuck_issue]

To overcome the stuck issues (above figure),  `Rover.stuck_count` was added to `Rover` class, which counts the consecutive frames the robot was stuck with no speed. If the `stuck_count` is larger than a threshold, which means the robot has got stuck for a while, then it will apply negative throttle to leave there.

To improve the directional stability, `Rover.front_nav_count` was added. It is the number of navigable pixels  in front of the robot. Only when it's less than a threshold, the robot makes a turn using the mean value of `Rover.nav_angles`.

### 2.4 Launching in Autonomous Mode   

The simulator was run at ***1024x768*** with ***good quality***. The rover can navigate and map autonomously with  **Mapped>50%** and **fidelity>60%**.

## 3. What's Next

- To increase the mapped area and reduce the searching time, I think the current searched/mapped area and the rover trajectory can be taken into consideration.  The idea is to make the rover not drive the same way twice.
- The pick & return function needs to be finished.
