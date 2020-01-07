import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from scripts.estimate_pose import estimate_single_pose
from scripts.common import pose_chain
from os import path, mkdir


# image loading functions ###
def load_custom_img(interpreter, img_dir, img_id):
    print("Loading image")
    img_path = '%s/%s' % (img_dir, img_id)  # custom image
    input_shape = interpreter.get_input_details()[0]['shape']
    img = load_img(img_path, target_size=input_shape[1:])  # maybe adjust target size to network?
    img_arr = img_to_array(img)
    img_arr = img_arr / 255.0  # transform to [0, 1] range.
    return img_arr


# pose estimation output functions ###
def test_image(interpreter, img, img_id):
    print("Running model")
    # Get input tensor for shape
    input_details = interpreter.get_input_details()

    # Compare required input shape with image shape, just to be sure.
    input_shape = input_details[0]['shape']
    img_data = tf.expand_dims(img, 0)
    print("Comparing input shape with image: ", input_shape, img_data.shape)

    # running the model and displaying results
    keypoint_data = estimate_single_pose(interpreter, img_data)
    print("Keypoints are computed")
    show_result(img, keypoint_data, img_id)


def show_result(img, keypoint_data, img_id):
    print("Showing results")
    # plot points:
    x_coords, y_coords, confidence_scores = keypoint_data
    plt.imshow(img, aspect='auto')
    plt.axis('off')
    plt.plot(x_coords, y_coords, 'o', markersize=4, markerfacecolor='c', markeredgecolor='c', markeredgewidth=2)

    # plot lines of the skeleton:
    for line in pose_chain:
        p1 = line[0]
        p2 = line[1]
        x_vals = [x_coords[p1], x_coords[p2]]
        y_vals = [y_coords[p1], y_coords[p2]]
        plt.plot(x_vals, y_vals, c='c')
    plt.tight_layout()

    # save output & show
    if not path.exists("./output/"):
        mkdir("./output/")
    out_name = "./output/{0}".format(img_id.replace(".jpg", ".png"))
    print("Saving output to:", out_name)
    plt.savefig(out_name, bbox_inches='tight', dpi=100, pad_inches=0.1)
    plt.show()


# main ###
def parse_args():
    parser = argparse.ArgumentParser(description='Predict a persons pose from an image with a single person.')
    parser.add_argument('--image_dir', type=str, default="./images",
                        help='The relative directory path that contains the image.')
    parser.add_argument('--image', type=str, default="ski.jpg",
                        help='The filename of the input image in the image directory.')
    args = parser.parse_args()
    return args


def main(arg):
    print("Loading model")
    # Load TFLite model and allocate tensors.
    model_dir = "./models/"
    model_path = model_dir + "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
    if not path.exists(model_path):
        print("Model does not exist. Downloading model...")
        if not path.exists(model_dir):
            mkdir(model_dir)
        url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
        model_file = model_dir + url.rsplit("/", 1)[1]
        filename = wget.download(url, out=model_file)
        print("Model saved to ", filename)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # test custom image
    img = load_custom_img(interpreter, arg.image_dir, arg.image)
    test_image(interpreter, img, arg.image)
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
