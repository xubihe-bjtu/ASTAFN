# ASTAFN

This is a PyTorch implementation of the paper: **ASTAFN: Bridging the Gap Between Weather Foundation Models and Accurate Station-Level Forecasting.**

## Requirements

The model is implemented using Python3 with dependencies specified in requirements.txt

## Usage

1. Install Python 3.9. For convenience, execute the following command.

~~~
pip install -r requirements.txt
~~~

2. **Prepare Data**: You can obtain the three well-preprocessed datasets—`California`, `GUANGDONG`, and `YUNNAN`—from the following [Google Drive](https://drive.google.com/drive/folders/115MqFuF6CqQm0c26Mou4CMXzrEwlPn05?usp=drive_link). After downloading, place the datasets in the `./dataset` folder.
3. To train and evaluate the model, simply execute the following examples within the `ASTAFN/` directory. The `target` parameter specifies the weather variable of interest: 0 for U-speed, 1 for V-speed, 2 for MSL, and 3 for TMP.

~~~
#GUANGDONG
python run_g.py --model ASTAFN --target 0

#YUNNAN
python run_y.py --model ASTAFN --target 0

#California
python run_c.py --model ASTAFN --target 0
~~~

