# color_grading_experiments

This repository lets you visualise how the image changes when we change the attributes. This is a pilot experiment where I have chosen the previously trained model.
I have devised it experiment to understand whether the idea of changing the attribute matrix changes the display image.

* `TODO`: Look into better encoding methods, by which we can actually interpret the attributes. This could be done with adding the complete image histogram alongside the neighbouring histogram.
* `TODO`: Reduce the number of attributes. Right now its 9 (for the pilot experiment).


# Some requirements

* `Pytorch` 
* `Torchvision` 
* `Pillow`, please do `pip install pillow`  
* `Streamlit`, please do `pip install streamlit`

You will also need to have the `lego_batman` movie yuv files:

* `../../video/lego_batman_4k/lego_batman_960x540_420_2020_10b.yuv`
* `../../video/lego_batman_hd/lego_batman_960x540_420_709_8b.yuv`


#  How to run the experiment

* `streamlit run inference.py -- --frame_index 300` 
This opens in the browser where we can use a slider to change the attributes.
You can change the frame of the lego_batman movie that you need to display. The deafult takes the frame 300.

Note: You need to use -- --frame_index to parse streamlit.

At every step on changing the attribute the inference runs on the GPU.

The starting point of attributes in the slider are the average values across all frames. These need not be the best values for that particular frame.
