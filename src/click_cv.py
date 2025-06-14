import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", img)

# Load your image
img = cv2.imread('img/turtlebot3_house_lvl2.png' )  # Change to your image path
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

# Loop until ESC key or window is closed
while True:
    key = cv2.waitKey(1) & 0xFF
    # ESC key to exit
    if key == 27:
        break
    # Detect if window is closed (optional but robust)
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
