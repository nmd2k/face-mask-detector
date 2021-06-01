# Face mask detector

<a href="https://hub.docker.com/r/manhdung20112000/face-mask"><img src="https://api.travis-ci.com/travis-ci/travis-web.svg?branch=master&status=passed" alt="Docker build"></a>

<a href="https://wandb.ai/nmd2000/Face_Mask"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg" alt="Visualize in WB"></a>

Table of contents
=================
* [Abstract](#abstract)
* [Training Result](#Training-Result)
* [Dataset](#Dataset)
* [Deployment](#Deployment)
* [Team member](#Team-member)
* [Reference](#Reference)

Abstract
========

Today is April 25, 2021, COVID-19 has affected countries all over the world. Turning back a few day ago, India has recorded approximately 2000 death case per day, which once again alert us about how dangerous this disease are.

In this project, our main purpose is to build a detection system, that able to detect a person is either wearing a mask. The system based on some YOLO [[1]](#1) version for object detection, they are:
- YOLOv3
- YOLOv3 fastest
- YOLOv5

By using the pre-defined models that were provided and supported by [ultralytics](https://github.com/ultralytics/). We will compare the result between these models, and implement a simple web application for run these model.

Training Result
==============

[assets_5]: https://github.com/ultralytics/yolov5/releases
[assets_3]: https://github.com/ultralytics/yolov3/releases

Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |mAP<sup>test<br>0.5:0.95 |mAP<sup>test<br>0.5 |Speed<br><sup>V100 (ms) | |params<br><sup>(M) 
---   |---                   |---                     |---                |---                      |---                |---                     |---|---              
[YOLOv5s][assets_5]    |640  |  65.4   |  93.1   |  65.4   |93.2     |**6.3**| |7.3   
[YOLOv5m][assets_5]    |640  |  66.5   |  93.9   |  66.7   |93.7     |7.9    | |21.4  
[YOLOv5l][assets_5]    |640  |**65.8** |**93.9** | 66.9   | 93.8     |12.1   | |47.0  
[YOLOv5x][assets_5]    |640  |  66.5   |  93.5   | **67.3**|**94.0** |  20.7  | |87.7  
| | | | | | || |
[YOLOv3 fastest][assets_3]   |640  | -       | -       | -       | -       | -       | | | - 
[YOLOv3-tiny][assets_3]      |640  |**55.7** |**87.9** |**55.6** |**87.8** |**3.4**  | |8.8  
[YOLOv3-SSP][assets_3]       |640  | -       | -       | -       | -       | -       | |63.0
[YOLOv3][assets_3]           |640  | -       | -       | -       | -       | -       | |61.9

We public our training result in [wandb](https://wandb.ai/) for if you want to dig deeper inside each model's training process, then make sure check out our project in [W&B](https://wandb.ai/nmd2000/Face_Mask).

Dataset
=======

The dataset is composed of [WIDER Face](http://shuoyang1213.me/WIDERFACE/) [[2]](#2) and [MAFA](www.escience.cn/people/geshiming/mafa.html) [[3]](#3). WIDER Face dataset contains **32,203** images with **393,703** normal faces which are refered as `non masked face`, MAFA contains **30,811** images with **35,806** `masked faces`.

![Dataset](result/the-dataset.png)

Due to the limition of computational power, we using the dataset composed by [AIZOOTech](https://github.com/AIZOOTech/FaceMaskDetection) which only contains **7959 images** in total, have been splited the dataset into 3 part: Train, Valid and Test; and converted them into YOLO format. You can find our dataset in [Kaggle](https://www.kaggle.com/nguyenmanhdung/facemaskyolo)

Or by running Kaggle API:
```
kaggle datasets download -d nguyenmanhdung/facemaskyolo
```

Deployment
==========

We've implemented a simple `Flask` application for demonstrate our work where located in `/deployment` folder. 

The quick demo is in the figure below, where we can see the `yolov3 tiny` model have a acceptable accuracy and a ablity of detecting multiple faces. 

![Result](result/result.png)

However, it's noticable some error that the model's made in some common scenarios in the video demonstration at Youtube:

<a href="https://youtu.be/YdGf6xMGzVQ?t=7" title=""><img src="result/video-cover.png" alt="Video demo" /></a>

To run the `Flask` application, direct to the `\deployment` folder, install all the requirements and run the following command:
```
$ pip install -r requirements.txt

$ python app.py
```

The output will be like shown below, thus, access to http://127.0.0.1:5000/ (or the port you have configured) to open the application(the browser should as for Webcam permission).

```
 * Serving Flask app "app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 219-729-123
```
You might notice that we also support an builed application through `Dockerfile`, which you can find at <a href="https://hub.docker.com/r/manhdung20112000/face-mask"><img src="https://api.travis-ci.com/travis-ci/travis-web.svg?branch=master&status=passed" alt="Docker build"></a>

*Note:* We still in process of developing this deployment, if you have anytrouble, feel free to contact us.

Team member
==========

**Dung Manh Nguyen (me)**
- Github: [manhdung20112000](https://github.com/manhdung20112000)
- Email: [manhdung20112000@gmail.com](manhdung20112000@gmail.com)

**Hai Phuc Nguyen**
- Github: [HaiNguyen2903](https://github.com/HaiNguyen2903)

**Hoang Huy Nguyen**
- Github: [hhoanguet](https://github.com/hhoanguet)

Reference
=================
<a id="1">[1]</a> Joseph Redmon et al.You Only Look Once: Unified, Real-Time Object Detection. 2016.arXiv:1506.02640 [cs.CV].

<a id="2">[2]</a> Shuo Yang et al. “WIDER FACE: A Face Detection Benchmark”. In:IEEE Conference onComputer Vision and Pattern Recognition (CVPR). 2016.

<a id="3">[3]</a> Adnane Cabani et al. “MaskedFace-Net–A dataset of correctly/incorrectly masked faceimages in the context of COVID-19”. In:Smart Health19 (2021), p. 100144.


