# Paper Implementation: Texture and exposure awareness based refill for HDRI reconstruction of saturated and occluded areas

> 
> This is the implementation for IET Image Processing paper ([Texture and exposure awareness based refill for HDRI reconstruction of saturated and occluded areas](https://doi.org/10.1049/ipr2.12257)). This paper proposed a pre-process algorithm to improve HDRI reconstruction.
> 
> 論文([Texture and exposure awareness based refill for HDRI reconstruction of saturated and occluded areas](https://doi.org/10.1049/ipr2.12257))の実装となります。この論文はHDR写真の合成を改善する前処理アルゴリズムを提案した。
> 

## Environment

For our implementation (下記の環境は実装に必要である):

- Python 3.6
- OpenCV for Python
- skimage >= 0.16
- scipy
- numpy

For optical flow tool: flownet2-tf, please check ([here](https://github.com/UncleJerry/flownet2-tf/blob/master/README.md))

For image refill tool, please check ([here](https://github.com/UncleJerry/Image_Completion/blob/master/code/README.md)). In a nutshell, this tool requires OpenCV for C++


## How to Obtain Experimental Data

| Which Data | Which Tool |
|:--|:--|
| Saturation Map for middle-exposed Image | saturation.py |
| Optical Flow Data | ([flownet2-tf](https://github.com/UncleJerry/flownet2-tf/blob/master/README.md)) |
| Refilled Image | ([im_complete_opencv_constraint.cpp](https://github.com/UncleJerry/Image_Completion/blob/master/code/README.md)) |

## Usage of Refill Tool (再補充ツールの使い方)

./ImageRefill Input_image Mask_image Restriction_image Output_name

## The Relation between Paper and Implementation (論文の章とソースコードの関連)

| Which Chapter　(どの章) | Which File (どのファイル) |
|:--|:--|
| Chapter 3.2 | P1.py |
| Chapter 3.3.2 | P2.py |
| Chapter 3.3.3 | P3.py |

Execution Order is P1.py --> P2.py --> refill tool --> P3.py.

Before execution, please prepare saturation map and optical flow data.

コードを実行する前に、saturation mapとoptical flow dataを用意しておいてください。

### About Optical Flow Data

You should use flownet2-tf to obtain two directions of motion data:

1. Lower-exposed ---> middle-exposed image
2. Middle-exposed ---> lower-exposed image

### About the Output of source codes (実行したら何を出力するか)

| Source Code | Output |
|:--|:--|
| P1.py | mask for low-exposed image which marks the areas need to be refilled |
| P2.py | small pieces of refill areas, and corresponding restriction |
| P3.py | Tile refill |

### Example of Refill (再補充された写真の)

Red box points out the result of refill tool basing on P2.py output. Blue boxes points out the tile result of P3.py.

Please refer to Scenes folder for more results.

![Refill　Result　Example](/RefillResultExample.jpg)