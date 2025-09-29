import cv2

#Geração do Vídeo

#cv2.VideoWriter(filename, fourcc, fps, frame_size, isColor)

#filename: Name of the output video file (e.g., “output.mp4”).
#fourcc: FourCC (Four Character Code) codec (e.g., “XVID”, “MJPG”, “MP4V”, “X264”).
#fps: Frames per second (e.g., 30.0).
#frame_size: Tuple specifying the frame width and height (e.g., (640, 480)).
#isColor: Boolean (True for color, False for grayscale).

openVideo = cv2.VideoCapture("C:/Users/joaop/Videos/input.mp4")

# Check if the video was opened successfully
if not openVideo.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get frame width and height
frame_width = int(openVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(openVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc =    cv2.VideoWriter_fourcc(*"MP4V") 
out = cv2.VideoWriter("output.mp4", fourcc, 30.0 (frame_height, frame_height))

while True:
    ret, frame = openVideo.read()
    if not ret:
        print("End of video error occurred.")
        
# Write the frame to the output video file
    out.write(frame)

    # Display the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Release everything
openVideo.release()
out.release()
cv2.destroyAllWindows()

''''
To write a video in OpenCV, we first open the video file using cv2.VideoCapture(). 
After successfully opening the video, we retrieve its properties, such as frame width and height, using cv2.CAP_PROP_FRAME_WIDTH and cv2.CAP_PROP_FRAME_HEIGHT. 
Next, we set up the video writer by defining the codec with cv2.VideoWriter_fourcc(), specifying the FPS and resolution. 
Inside a loop, we continuously read frames using cap.read(), write each frame to the output video using out.write(frame), and display it using cv2.imshow(). 
The loop continues until the video ends (ret == False) or the user presses the ‘q’ key to exit. 

Finally, we release the video objects with cap.release() and out.release(), ensuring all resources are freed, and close all OpenCV windows using cv2.destroyAllWindows().
'''