# import the opencv library
import cv2

import time
from model import HandmarkModel, Classifier


handmarks_model = HandmarkModel()

clf = Classifier()
print("start")
# define a video capture object
vid = cv2.VideoCapture(0)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

while vid.isOpened():
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    if frame is None:
        print("Breaking")
        break
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = handmarks_model.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    handmarks_model.draw_handmarks(image, results)

    output, prob = clf.predict(results)
    print(output)

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(
        image,
        str(int(fps))
        + " FPS "
        + (output + " " + (str(int(prob * 100))) if output else ""),
        (10, 70),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Display the resulting frame
    cv2.imshow("frame", image)

    # use 'q' key to close the program
    if cv2.waitKey(1) == ord("q"):
        break

# After the loop release the vid object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
