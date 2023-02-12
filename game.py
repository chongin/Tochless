import cv2
import mediapipe as mp
import random
import threading, time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

robot_shape = random.choice(['rock','paper','scissor'])
check = True

def update_shape():
  global robot_shape, check
  result = "thinking"
  time.sleep(2)
  robot_shape = random.choice(['rock','paper','scissor'])
  print(robot_shape)
  check = True

# x = threading.Thread(target=update_shape)
# x.start()

def dectect_fingers(handLandmarks):
  fingerCount=0
  if handLandmarks[4][0] > handLandmarks[3][0]: #thumb finger
    fingerCount = fingerCount+1
  if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
    fingerCount = fingerCount+1
  if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
    fingerCount = fingerCount+1
  if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
    fingerCount = fingerCount+1
  if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
    fingerCount = fingerCount+1
  return fingerCount

def dectect_shape_by_finger_count(fingerCount):
  shape=""
  if fingerCount == 5:
    shape="paper"
  elif fingerCount == 0:
    shape="rock" 
  elif fingerCount == 2:
    shape="scissor"
  else:
    shape="Wrong"
  return shape

def dectect_handshade(results):
  left_hand_shape=""
  right_hand_shape=""
  for hand_landmarks in results.multi_hand_landmarks:
      handIndex = results.multi_hand_landmarks.index(hand_landmarks)
      handLabel = results.multi_handedness[handIndex].classification[0].label
      handLandmarks = []
      for landmarks in hand_landmarks.landmark:
        handLandmarks.append([landmarks.x, landmarks.y])
      
      if handLabel == "Left":
        fingerCount = dectect_fingers(handLandmarks)
        left_hand_shape=dectect_shape_by_finger_count(fingerCount)
      else:
        fingerCount = dectect_fingers(handLandmarks)
        right_hand_shape=dectect_shape_by_finger_count(fingerCount)

      mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

  return (left_hand_shape, right_hand_shape)


result = "waiting"

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
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initially set finger count to 0 for each cap
    fingerCount = 0
    if results.multi_hand_landmarks:
      shapes=dectect_handshade(results)
     #robot_shape=random.choice(['rock','paper','scissor'])
      if check:
        if shapes[0] == "paper":
          if robot_shape == "paper":
            result="Draw"
          elif robot_shape == "rock":
            result="You Win"
          elif robot_shape == "scissor":
            result="Computer Wins"
          else:
            result="Waiting"

        if shapes[0] == "rock":
          if robot_shape == "paper":
            result="Computer Wins"
          elif robot_shape == "rock":
            result="Draw"
          elif robot_shape == "scissor":
            result="You Win"
          else:
            result="Waiting"
        if shapes[0] == "scissor":
          if robot_shape == "paper":
            result="Computer Wins"
          elif robot_shape == "rock":
            result="You Win"
          elif robot_shape == "scissor":
            result="Draw"
          else:
            result="Waiting"
        check = False
        x = threading.Thread(target=update_shape)
        x.start()

    # Display finger count
    cv2.putText(image, str(result), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # Display image
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()