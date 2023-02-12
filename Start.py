import cv2, math
import mediapipe as mp
from pynput.mouse import Button, Controller

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

mouse = Controller()

def find_pos(result,img):
  ls = []

  if result.multi_hand_landmarks:
    myhand = results.multi_hand_landmarks[0]
    for id, lm in enumerate(myhand.landmark):
      h, w, c = img.shape
      cx, cy = int(lm.x * w), int(lm.y * h)
      ls.append([id, cx, cy])

  return ls

def find_dis(a,b):
  dis = math.dist(a,b)
  return dis

def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)

cap = cv2.VideoCapture(0)
ls = []
dis = None
font = cv2.FONT_HERSHEY_SIMPLEX
click = "hold"


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    ls = find_pos(results,image)

    if ls:
      cv2.circle(image,(ls[8][1],ls[8][2]),50,(255,255,255),5)
      cv2.circle(image,(ls[4][1],ls[4][2]),50,(255,255,255),5)
      cv2.line(image,(ls[8][1],ls[8][2]),(ls[4][1],ls[4][2]),(255,0,255),5)

      dis = find_dis((ls[8][1],ls[8][2]),(ls[4][1],ls[4][2]))

      if ls[20][1] > ls[17][1]:
        click = "click"
      else:
        click = "hold"

      if dis < 40:
        if click == "click":
          mouse.click(Button.left, 2)
        else:
          mouse.press(Button.left)
      else:
        mouse.release(Button.left)

      center_point=midpoint(ls[8][1], ls[8][2], ls[4][1],ls[4][2])
      mouse.position = (center_point[0], center_point[1])

    cv2.putText(image,str(click),(50,50),font,1,(255,255,0),2,cv2.LINE_AA)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
