# IET Image Processing: Texture and exposure awareness based refill for HDRI reconstruction of saturated and occluded areas

> 
> This is the implementation for paper ([Texture and exposure awareness based refill for HDRI reconstruction of saturated and occluded areas](https://doi.org/10.1049/ipr2.12257))
> 
> 論文([Texture and exposure awareness based refill for HDRI reconstruction of saturated and occluded areas](https://doi.org/10.1049/ipr2.12257))の実装となります。
> 日本語は英語の後に続きます。（工事中）

## Environment

For our implementation:
- Python 3.6
- OpenCV for Python
- skimage >= 0.16
- scipy
- numpy

For flownet2-tf, please check ([here](https://github.com/UncleJerry/flownet2-tf/blob/master/README.md))

For image refill tool, please check ([here](https://github.com/UncleJerry/Image_Completion/blob/master/code/README.md)). In a nutshell, this tool requires OpenCV for C++


## How to Get Experimental Data

| Which Data | Which Tool |
|:--|:--|
| Saturation Map for middle-exposed Image | saturation.py |
| Optical Flow Data | ([flownet2-tf](https://github.com/UncleJerry/flownet2-tf/blob/master/README.md)) |
| Refilled Image | ([im_complete_opencv_constraint.cpp](https://github.com/UncleJerry/Image_Completion/blob/master/code/README.md)) |

## Usage of Refill Tool

./ImageRefill Input_image Mask_image Restriction_image Output_name

## The Relation between Paper and Implementation

| Which Chapter | Which File |
|:--|:--|
| Chapter 3.2 | P1.py |
| Chapter 3.3.2 | P2.py |
| Chapter 3.3.3 | P3.py |

Execution Order is P1.py --> P2.py --> refill tool --> P3.py.

Before execution, please prepare saturation map and optical flow data.

### About Optical Flow Data

Both lower-exposed ---> middle-exposed image and middle-exposed ---> lower-exposed image data is required by the research.

### About the Output of source codes

| Source Code | Output |
|:--|:--|
| P1.py | mask for low-exposed image which marks the areas need to be refilled |
| P2.py | small pieces of refill areas, and corresponding restriction |
| P3.py | Tile refill |

### Example of Refill

Red box points out the result of refill tool basing on P2.py output. Blue boxes points out the tile result of P3.py.

Please refer to Scenes folder for more results.