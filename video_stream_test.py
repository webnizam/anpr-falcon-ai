import cv2

capture_device = 'rtsp://192.168.1.50/live0'
# capture_device = 1


cap = cv2.VideoCapture(capture_device)

while(True):
    ret, frame = cap.read()
    cv2.imshow('Test Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
