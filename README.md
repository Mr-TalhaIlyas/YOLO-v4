# YOLO v4

This repo exaplins how to train [YOLOv4](https://arxiv.org/abs/2004.10934) model on your custom dataset. 

## Dependencies

Some of the main requirements are
```
pytorch
```

## Roboflow

I'll be using the [YOLOv4](https://github.com/roboflow-ai/pytorch-YOLOv4) repo in this tutorial, and go through the steps of how 
Original [Colab Notebook](https://colab.research.google.com/drive/1b08y_nUYv5UtDY211NFfINY7Hy_pgZDt#scrollTo=lAoLxBEEz4FW)

1. Prepare your data 
2. Convert data format from PASCAl_VOC to YOLO-v4 PyTorch format
3. Installing repo


## Dataset Preparation

First to train an object detection model you need a dataset annotated in proper format so download publically available datasets from [here](https://public.roboflow.com/).
I'd recommend starting by downloading already available dataset. There are alot of format options available in Roboflow but for this repo we need `YOLO v4 PyTorch` as this 

![alt text](https://github.com/Mr-TalhaIlyas/YOLO-v4/blob/master/screens/img.png)

or you can also make you own dataset using `labelimg`. A full tutorial for that is [here](https://github.com/tzutalin/labelImg)
The ouput annotation file for label me is `.xml` format but our yolov4 model can't read that so we need to convert the dataset into proper format

## Dataset Format Conversion
 After labelling the data put both img files and `.xml` files in the same dir.
and run the `voc2yolo.py` file from scripts

In the first few lines of the `.py` file change following lines accordingly

```python
VOC_class_names = [ "Platelets", "RBC","WBC"] # class names
 
images_dir = 'path/to/data/dir/train/'
op_dir = 'path/to/data/dir/yolo_format/'

```
After that make your data dir like following

```
📦cells
 ┣ 📂data
 ┃ ┣ 📜train.txt
 ┃ ┗ 📜val.txt
 ┣ 📂test
 ┃ ┣ 📜BloodImage_00086_jpg
 ┃ ┣ 📜BloodImage_00092_jpg
 ┃ ┣ 📜_annotations.txt
 ┃ ┗ 📜_classes.txt
 ┣ 📂train
 ┃ ┣ 📜BloodImage_00086_jpg
 ┃ ┣ 📜BloodImage_00092_jpg
 ┃ ┣ 📜_annotations.txt
 ┃ ┗ 📜_classes.txt
 ┗ 📂valid
 ┃ ┣ 📜BloodImage_00086_jpg
 ┃ ┣ 📜BloodImage_00092_jpg
 ┃ ┣ 📜_annotations.txt
 ┃ ┗ 📜_classes.txt
```
For details on data `dir` structure you can see `sample_data` dir provided in repo.

## Installation
Getting started wiht the installation make a new conda `env`

```
conda create -n yolo4 python=3.7.6
```
activate `env` and `cd` to a `dir` where you will keep all your `data` and `scripts`.

### Scaled YOLO v4
Now form inside the `scirpts` dir copy the `jupyter notebooks` and place in the same `dir` you choose first.

Now run the notebook `setup_YOLOv4.ipynb` ***sequentelly**
Place your data inside the `dir` you choose
Now run the `Scaled_YOLOv4.ipynd` follow steps inside notebook


