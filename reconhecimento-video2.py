import cv2
import face_recognition
import time

video_path = "C:/Users/07443204370/Downloads/short.mp4"
video_capture = cv2.VideoCapture(video_path)

scale_factor = 0.8

fps = video_capture.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None or fps != fps:
    fps = 25

total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps

print(f"FPS detectado: {fps}, total frames: {total_frames}, duração real: {video_duration:.2f}s")

start_time = time.time()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)

    scale = 1 / scale_factor
    face_locations = [(
        int(top*scale),
        int(right*scale),
        int(bottom*scale),
        int(left*scale)
    ) for (top, right, bottom, left) in face_locations]

    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('Reconhecimento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
inspection_time = end_time - start_time

print(f"Tempo de inspeção (busca por rostos): {inspection_time:.2f} segundos")
print(f"Duração original do vídeo: {video_duration:.2f} segundos")

video_capture.release()
cv2.destroyAllWindows()
