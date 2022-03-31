# What does this repository contain?
This repository contains example code for training, quantizing and compiling yolov4 and yolov4 tiny model with tensorflow2 and Vitis-AI 2.0.
# How to use this repository?
The yolov4 model is defined in yolov4_final.py, the yolov4-tiny model is defined in yolov4-tiny-tf2/yolo_output_model.py file. To adapt the network structure, please modify these two files.

To train the model, please modify the yolov4_train.py file accordingly.

To quantize the model, establish the necessary docker environment according the following link: https://www.hackster.io/AlbertaBeef/vitis-ai-2-0-flow-for-avnet-vitis-platforms-06cfd6. Then copy the resulting .pb directory, the yolov4_quantize.py and the custom.json from the repository https://github.com/wwMark/u96v2_vitis_ai_auto_qp_flow to the container in the same directory, and run:

python yolov4_quantize.py.

To compile the quantized model, run:

vai_c_tensorflow2 -m ./quantized_yolov4_model.h5 -a ./custom.json -o ./compiled_yolov4_model -n yolov4

The resulting .xmodel file can then be copied onto the u96v2 board. To use it on the board, feed the path of the .xmodel as an argument when running the example application.