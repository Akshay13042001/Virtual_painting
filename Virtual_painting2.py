import mediapipe as mp
import cv2
import numpy as np
import time

# Constants
ml = 150
max_x, max_y = 250 + ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0

# Function to get tools
def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    else:
        return "erase"

# Function to check if index is raised
def index_raised(yi, y9):
    return (y9 - yi) > 40

# Load tools image
tools = cv2.imread(r'F:\Deep Learning Projects\MediaPipe Projects\tools.png')
if tools is None:
    print("Error: Unable to read the tools image.")
    exit()

# Convert tools to uint8 if successfully loaded
if not isinstance(tools, np.ndarray):
    print("Error: Unable to convert tools image to uint8.")
    exit()

tools = tools.astype('uint8')

# Create mask
mask = np.ones((480, 640)) * 255
mask = mask.astype('uint8')

# Hands
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Create a whiteboard canvas
whiteboard = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Main loop
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Your current tool is set to:", curr_tool)
                    time_init = True
                    rad = 40

            else:
                time_init = True
                rad = 40

            if curr_tool == "draw":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                    cv2.line(whiteboard, (prevx, prevy), (x, y), (0, 0, 0), thick)
                    prevx, prevy = x, y

                else:
                    prevx = x
                    prevy = y

            elif curr_tool == "line":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not(var_inits):
                        xii, yii = x, y
                        var_inits = True

                    cv2.line(frm, (xii, yii), (x, y), (50, 152, 255), thick)
                    
                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), 0, thick)
                        cv2.line(whiteboard, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "circle":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not(var_inits):
                        xii, yii = x, y
                        var_inits = True

                    cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (255, 255, 0), thick)
                    
                else:
                    if var_inits:
                        cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (0, 255, 0), thick)
                        cv2.circle(whiteboard, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (0, 255, 0), thick)
                        var_inits = False

            elif curr_tool == "rectangle":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not(var_inits):
                        xii, yii = x, y
                        var_inits = True

                    cv2.rectangle(frm, (xii, yii), (x, y), (0, 255, 255), thick)
                    
                    
                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                        cv2.rectangle(whiteboard, (xii, yii), (x, y), (0, 255, 255), thick)
                        var_inits = False

            elif curr_tool == "erase":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                    cv2.circle(mask, (x, y), 30, 255, -1)
                    cv2.circle(whiteboard, (x, y), 30,(255,255,255), -1)

    op = cv2.bitwise_and(frm, frm, mask=mask)
    frm[:, :, 1] = op[:, :, 1]
    frm[:, :, 2] = op[:, :, 2]

    frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

    cv2.putText(frm, curr_tool, (270 + ml, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Display the whiteboard drawing
    cv2.imshow("Whiteboard", whiteboard)

    cv2.imshow("Paint App", frm)
    key = cv2.waitKey(1) & 0xFF

    # Save the whiteboard drawing when 's' key is pressed
    if key == ord('s'):
        # Generate a unique filename based on the current timestamp
        timestamp = time.strftime("%Y%m%d%H%M%S")
        file_name = f"whiteboard_drawing_{timestamp}.png"
        cv2.imwrite(file_name, whiteboard)
        print(f"Whiteboard drawing saved as {file_name}")

    # Break the loop if q key is pressed
    elif key == ord('q'):
        break

cv2.destroyAllWindows()    
