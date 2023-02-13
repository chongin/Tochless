import cv2
import mediapipe as mp
import threading, time
from enum import Enum

#hand shape
ROCK = "Rock"
PAPER = "Paper"
SCISSORS = "Scissors"
NOSHAPE = "Noshape"

#game status
TIE = "Tie"
FIRSTHAND = "First Hand Win"
SECONDHAND = "Second Hand Win"
LEFTHAND = "Left Hand Win"
RIGHTHAND = "Right Hand Win"

WAITTING = "Waiting"
DETECTING = "Detecting"

class HandShape:
  hand_label = ""
  landmarks = []
  _shapes = NOSHAPE

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
    self._detect_fingers()
    print("hand:{0} finger count={1}".format(self.hand_label, self._finger_count))
    if self._finger_count == 5:
      self._shapes = PAPER
    elif self._finger_count == 0:
      self._shapes = ROCK
    elif self._finger_count == 2 and self._middle_finger and self._index_finger:
      self._shapes = SCISSORS
    else:
      self._shapes = NOSHAPE

  def get_shape(self):
    return self._shapes

  def get_label(self):
    if self.hand_label == "Left":
      return RIGHTHAND #system already oppsited hand.
    elif self.hand_label == "Right":
      return LEFTHAND
    else:
      return "Unknow Hand"


  def _detect_fingers(self):
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
  
class Game:
  handshapes=[]
  def __init__(self, handshapes):
    self.handshapes = handshapes

  def calc_winner(self):
    if len(self.handshapes) != 2:
      return DETECTING

    first_hand = self.handshapes[0]
    second_hand = self.handshapes[1]
    print("first_hand shape:{0}, second_hand:shape:{1}".format(first_hand.get_shape(), second_hand.get_shape()))
    if first_hand.get_shape() == NOSHAPE or second_hand.get_shape() == NOSHAPE:
        return DETECTING

    if first_hand.get_shape() == second_hand.get_shape():
      return TIE
    elif first_hand.get_shape() == PAPER and second_hand.get_shape() == ROCK:
      if first_hand.get_label() == second_hand.get_label():
        return FIRSTHAND
      else:
        return first_hand.get_label()
    elif first_hand.get_shape() == ROCK and second_hand.get_shape() == SCISSORS:
      if first_hand.get_label() == second_hand.get_label():
        return FIRSTHAND
      else:
        return first_hand.get_label()
    elif first_hand.get_shape() == SCISSORS and second_hand.get_shape() == PAPER:
      if first_hand.get_label() == second_hand.get_label():
        return FIRSTHAND
      else:
        return first_hand.get_label()
    else:
      if first_hand.get_label() == second_hand.get_label():
        return SECONDHAND
      else:
        return second_hand.get_label()


#global varible
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
font_scale = 2
font_thickness = 5

pTime = 0
game_status = WAITTING
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

    image.flags.writeable = False #False: To improve performance
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True #Add some text to the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      handshapes=[]
      for hand_landmarks in results.multi_hand_landmarks:
        hand_index = results.multi_hand_landmarks.index(hand_landmarks)
        handl_label = results.multi_handedness[hand_index].classification[0].label
        landmarks = []
        for landmark in hand_landmarks.landmark:
          landmarks.append([landmark.x, landmark.y])

        #create HandShape instance
        handshape = HandShape(handl_label, landmarks)
        handshape.detect_shape()
        handshapes.append(handshape)
        
        # Display hand landmarks
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


       # Display hand shape
      if handshapes[0]:
        shape = handshapes[0].get_shape()
        cv2.putText(image, str(shape), (50, 70), font, font_scale, (0,0,255), font_thickness)
      if len(handshapes) == 2:
        shape = handshapes[1].get_shape()
        cv2.putText(image, str(shape), (900, 70), font, font_scale, (0,255,0), font_thickness)

      #calc the game status
      game = Game(handshapes)
      game_status = game.calc_winner()
      print("Result: {0}, handshapes: {1}".format(game_status, len(handshapes)))
    else:
      game_status = WAITTING

    # Display game status
    cv2.putText(image, str(game_status), (50, 400), font, font_scale, (255,0,0), font_thickness)

    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (480, 70), font, font_scale, (255,0,0), font_thickness)

    cv2.imshow('Paper Rock Scissors', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()