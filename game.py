import cv2
import mediapipe as mp
import random
import threading, time
from enum import Enum

ROCK = "rock"
PAPER = "paper"
SCISSORS = "scissors"
NOSHAPE = "no_shape"

class Winners(Enum):
  Tie = 1
  FirstHand = 2
  SecondHand = 3
  NoWinner = 4

class Handshape:
  hand_label = ""
  landmarks = []
  shape=NOSHAPE

  _thumb_finger = False
  _index_finger = False
  _middle_finger = False
  _ring_finger = False
  _pinky_finger = False
  _finger_count = 0
  
  def __init__(self,hand_label, landmarks):
    self.hand_label = hand_label
    self.landmarks = landmarks

  def detect_shape(self):
    #cacl thumb finger
    if self.hand_label == "Left":
      if self.landmarks[4][0] > self.landmarks[3][0]:
        self._finger_count += 1
        self._thumb_finger = True
    else: #Right Hand
      if self.landmarks[4][0] < self.landmarks[3][0]:
        self._finger_count += 1
        self._thumb_finger = True

    #calc other finger
    if self.landmarks[8][1] < self.landmarks[6][1]:
      self._finger_count += 1
      self._index_finger = True
    if self.landmarks[12][1] < self.landmarks[10][1]:
      self._finger_count += 1
      self._middle_finger = True
    if self.landmarks[16][1] < self.landmarks[14][1]:
      self._finger_count += 1
      self._ring_finger = True
    if self.landmarks[20][1] < self.landmarks[18][1]:
      self._finger_count += 1
      self._pinky_finger = True

    self._calc_shape()
  
  def _calc_shape(self):
    print("hand:{0} finger count={1}".format(self.hand_label, self._finger_count))
    if self._finger_count == 5:
      self.shape = PAPER
    elif self._finger_count == 0:
      self.shape = ROCK
    elif self._finger_count == 2 and self._middle_finger and self._index_finger:
      self.shape = SCISSORS
    else:
      self.shape = NOSHAPE
    

#not only allow left, right hand to play, but also allow left left, right right
def calc_winner(handshapes):
  if len(handshapes) != 2:
    return Winners.NoWinner

  hand1 = handshapes[0]
  hand2 = handshapes[1]

  print("hand1 shape:{0}, hand2:shape:{1}".format(hand1.shape, hand2.shape))
  if hand1.shape == hand2.shape:
    return Winners.Tie
  elif hand1.shape == PAPER and hand2.shape == ROCK:
    return Winners.FirstHand
  elif hand1.shape == ROCK and hand2.shape == SCISSORS:
    return Winners.FirstHand
  elif hand1.shape == SCISSORS and hand2.shape == PAPER:
    return Winners.FirstHand
  else:
    if hand1.shape == NOSHAPE or hand2.shape == NOSHAPE:
      return Winners.NoWinner
    else:
      return Winners.SecondHand

def dectect_handshade(results):
  handshapes=[]
  for hand_landmarks in results.multi_hand_landmarks:
      hand_index = results.multi_hand_landmarks.index(hand_landmarks)
      handl_label = results.multi_handedness[hand_index].classification[0].label
      landmarks = []
      for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y])

      handshape = Handshape(handl_label, landmarks)
      handshape.detect_shape()
      handshapes.append(handshape)
      
      mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

  return handshapes


result = "waiting"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # To improve performance, optionally mark the image as not writeable to
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      handshapes = dectect_handshade(results)
      result = calc_winner(handshapes)
      

    # Display finger count
    cv2.putText(image, str(result), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # Display image
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()