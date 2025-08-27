import cv2
import face_recognition as fr

video = cv2.VideoCapture("/home/afranio/Downloads/pessoas-andando.mp4")

fps = video.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25
delay = int(1000 / fps)

frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1

    display_frame = frame.copy()

    if frame_count % 3 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = fr.face_locations(rgb_small, number_of_times_to_upsample=0, model="hog")
        face_encodings = fr.face_encodings(rgb_small, face_locations)

        scale = 1
        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

            print("\n=== Rosto detectado ===")
            print(encoding)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", display_frame)
    cv2.resizeWindow("Frame", 900, 700)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
