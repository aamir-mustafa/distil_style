# color_grading_experiments

This repository lets you visualise how the image changes when we change the attributes. This is a pilot experiment where I have chosen the previously trained model.
I have devised it experiment to understand whether the idea of changing the attribute matrix changes the display image.

* `TODO`: Look into better encoding methods, by which we can actually interpret the attributes.
* `TODO`: Reduce the number of sttributes. right now its 9 (for the pilot experiment).


# Some requirements

* `Pytorch` 
* `Torchvision` 
* `Streamlit`, please do pip install streamlit

You will also need to have the `lego_batman` movie yuv files:

* `../../video/lego_batman_4k/lego_batman_960x540_420_2020_10b.yub`
* `../../video/lego_batman_hd/lego_batman_960x540_420_709_8b.yub`


#  How to run the experiment

* `streamlit run inference.py -- --frame_index` 
This opens in the browser where we can use a slider to change the attributes.
You can change the frame of the lego-batman movie that you need to display. The deafult takes the frame 300.
Note: You need to use -- --frame_index to parse streamlit.
