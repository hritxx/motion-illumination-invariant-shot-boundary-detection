import cv2
import os


def extract_all_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{count:05d}.jpg"), frame)
        count += 1
    cap.release()
    return count


video_path = 'path_to_the_desired_video_file'
output_dir = 'name_of_output_directory'
total_frames = extract_all_frames(video_path, output_dir)
print(total_frames)