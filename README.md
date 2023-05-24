# Setup env mmdetection

```

conda create --name demo_DIU python=3.8 -y
conda activate demo_DIU
conda install pytorch torchvision -c pytorch # On GPU platforms

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

mim install mmdet

git clone https://github.com/truong11062002/page_object_detection_streamlit.git
cd page_object_detection_streamlit

```

# Setup env streamlit
```
pip install streamlit
```

# Tree folder
Ensuring the right data tree format

    page_object_detection_streamlit
    ├── configs
    │   ├── config1.py
    |   ├── config2.py
    |   ├── config3.py
    |   ├── ...
    ├── models
    │   ├── model1.pth
    |   ├── model1.pth
    |   ├── model1.pth
    |   ├── ...
    ├── demo_DIU.py
Please download model at this link: [model](https://github.com/truong11062002/page_object_detection_streamlit/releases/download/model/truongDIU.pth)

**configs**: containing config for each method

**models**: containing model after training

# Run demo on streamlit
```
streamlit run demo_DIU.py
```
# Upload an image to predict

![](https://hackmd.io/_uploads/H191o-jHn.png)

# Result after predicting
Input             |  Output
:-------------------------:|:-------------------------:
![](https://hackmd.io/_uploads/B1ius-oS2.jpg)  |  ![](https://hackmd.io/_uploads/rJQKs-sHn.jpg)
