import cv2

def create_summary(video_path, scores, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    summary_frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if scores[i] > threshold:
            summary_frames.append(frame)
        i += 1
    cap.release()

    # Save the summary video
    out = cv2.VideoWriter('outputs/summary.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))
    for frame in summary_frames:
        out.write(frame)
    out.release()
