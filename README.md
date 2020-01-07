Result from a self-imposed 48h mini-hackathon in order to explore pose estimation. The main script reads the pretrained TFLite PoseNet and applies inference on a single image. While the model can handle multiple persons, the current system assumes one person per image. The input image is (down)scaled to [257, 257], the model's default input. The supplied images are from the COCO 2017 validation set (http://cocodataset.org/). 

Example usage: `python single_pose_predict.py --image tennis.jpg`

More information on the model can be found [at Google](https://www.tensorflow.org/lite/models/pose_estimation/overview). This [Medium article](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5) provides more information on PoseNet in general.
