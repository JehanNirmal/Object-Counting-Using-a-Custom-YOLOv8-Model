import os
import cv2
from ultralytics import YOLO, solutions

model_path = r"C:\Users\USER\Documents\yolo 8\venv\best.pt"
video_path = r"C:\Users\USER\Documents\yolo 8\venv\pack.mp4"
output_video_path = "object_counting_output.avi"


model = YOLO(model_path)


cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_x = 2400  
line_points = [(line_x, 0), (line_x, h)] 


video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
    view_in_counts=True,  
    view_out_counts=True,  
    count_txt_color=(0, 0, 255),  
    count_bg_color=(255, 255, 255)  
)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 4
font_thickness = 3
font_color = (255, 255, 255)  

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)
    
    video_writer.write(im0)

    cv2.imshow('Object Counting', im0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
