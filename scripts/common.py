from enum import IntEnum, unique
from numpy import exp

@unique
class BodyPart(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


pose_chain = [
  [BodyPart.NOSE, BodyPart.LEFT_EYE], [BodyPart.LEFT_EYE, BodyPart.LEFT_EAR], [BodyPart.NOSE, BodyPart.RIGHT_EYE],
  [BodyPart.RIGHT_EYE, BodyPart.RIGHT_EAR],

  [BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW], [BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST],
  [BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP], [BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE],
  [BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE],
    
  [BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER],

  [BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW], [BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST],
  [BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP], [BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE],
  [BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE]
]


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


