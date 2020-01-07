import tensorflow as tf
from scripts.common import *

output_stride = 32


# find_keypoints computes the maximum values of the heatmap for every body part
def find_keypoints(hm_data):
    height, width, n_keypoints = hm_data.shape
    print("heatmap shape:", hm_data.shape)
    keypoint_positions = []
    for keypoint in range(n_keypoints):
        max_val = hm_data[0, 0, keypoint]
        max_row = 0
        max_col = 0
        for row in range(height):
            for col in range(width):
                val = hm_data[row, col, keypoint]
                if val > max_val:
                    max_row = row
                    max_col = col
                    max_val = val
                    # print("{0} > {1}".format(val, max_val))
        keypoint_positions.append([max_row, max_col])
    print("keypoint positions found")
    # print(keypoint_positions)
    return keypoint_positions


# find_coords converts the found keypoints into coordinates in the image.
def find_coords(key_positions, offsets, hm_data, img_shape):
    hm_height, hm_width, num_keypoints = hm_data.shape
    img_height = img_shape[1]
    img_width = img_shape[2]
    x_coords = []
    y_coords = []
    confidence_scores = []
    for idx, position in enumerate(key_positions):
        posY = position[0]
        posX = position[1]
        y_coord = int(posY * output_stride + offsets[posY][posX][idx])
        x_coord = int(posX * output_stride + offsets[posY][posX][idx + num_keypoints])
        # y_coord = int(position[0] / float(hm_height - 1) * img_height + offsets[posY][posX][idx])
        # x_coord = int(position[1] / float(hm_width - 1) * img_width + offsets[posY][posX][idx + num_keypoints])
        conf = sigmoid(hm_data[posY][posX][idx])
        x_coords.append(x_coord)
        y_coords.append(y_coord)
        confidence_scores.append(conf)
    print("confidence: ", sum(confidence_scores))
    return x_coords, y_coords, confidence_scores


# estimate_single_pose computes the pose of a person, assuming that there is only one person in the image.
def estimate_single_pose(interpreter, img_data):
    # get the names of the required tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # present input to the model
    interpreter.set_tensor(input_details[0]['index'], img_data)

    # run model / compute output
    interpreter.invoke()

    # collect output
    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = output_details
    hm_data = tf.squeeze(interpreter.get_tensor(heatmaps_result['index']))
    offset_data = tf.squeeze(interpreter.get_tensor(offsets_result['index']))

    # process output
    keypoint_positions = find_keypoints(hm_data)
    x_coords, y_coords, confidence_scores = find_coords(keypoint_positions, offset_data, hm_data, img_data.shape)
    # print(x_coords, y_coords, confidence_scores)
    return x_coords, y_coords, confidence_scores
