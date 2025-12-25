import cv2

def extract_and_resize_frames(
    video_path: str,
    target_width: int,
    frames_per_second: int = 1
):
    """
    Extract frames from a video at a fixed rate and resize them so that
    their width equals target_width (aspect ratio preserved).

    Args:
        video_path (str): Path to the video file.
        target_width (int): Desired width of output frames.
        frames_per_second (int): Number of frames to extract per second.

    Returns:
        list: List of resized frames as NumPy arrays (BGR format).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / frames_per_second))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            h, w = frame.shape[:2]
            scale = target_width / w
            target_height = int(h * scale)

            resized = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA
            )
            frames.append(resized)

        frame_idx += 1

    cap.release()
    return frames
