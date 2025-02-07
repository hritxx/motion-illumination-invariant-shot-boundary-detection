import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import chisquare
import os
from sklearn.cluster import KMeans
from scipy.signal import find_peaks


def resize_frame(frame, width=640):
    height = int(frame.shape[0] * (width / frame.shape[1]))
    return cv2.resize(frame, (width, height))


def denoise_frame(frame, kernel_size=(5, 5)):
    return cv2.GaussianBlur(frame, kernel_size, 0)


def equalize_histogram(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def temporal_smooth(frames, window_size=3):
    smoothed = []
    for i in range(len(frames)):
        start = max(0, i - window_size // 2)
        end = min(len(frames), i + window_size // 2 + 1)
        smoothed.append(np.mean(frames[start:end], axis=0).astype(np.uint8))
    return smoothed


def load_and_preprocess_frames(frames_folder):
    frames = []
    for filename in sorted(os.listdir(frames_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            frame_path = os.path.join(frames_folder, filename)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame = resize_frame(frame)
                frame = denoise_frame(frame)
                frame = equalize_histogram(frame)
                frames.append(frame)
    return temporal_smooth(frames)


def compute_lbp(image, P=8, R=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    return hist.astype("float")


def lab_difference(frame1, frame2):
    lab1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2LAB)
    diff = np.mean(np.abs(lab1.astype("float") - lab2.astype("float")))
    return diff


def optical_flow_difference(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)


def canny_edge_difference(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)
    diff = np.sum(np.abs(edges1.astype("float") - edges2.astype("float")))
    return diff


def detect_changes(frames_folder):
    frames = load_and_preprocess_frames(frames_folder)
    lbp_diffs = []
    lab_diffs = []
    flow_diffs = []
    edge_diffs = []

    epsilon = 1e-10 

    for i in range(1, len(frames)):
        lbp_prev = compute_lbp(frames[i - 1])
        lbp_curr = compute_lbp(frames[i])

        lbp_prev += epsilon
        lbp_curr += epsilon

        lbp_diff = chisquare(lbp_prev, lbp_curr)[0]
        lbp_diffs.append(lbp_diff)

        lab_diff = lab_difference(frames[i - 1], frames[i])
        lab_diffs.append(lab_diff)

        flow_diff = optical_flow_difference(frames[i - 1], frames[i])
        flow_diffs.append(flow_diff)

        edge_diff = canny_edge_difference(frames[i - 1], frames[i])
        edge_diffs.append(edge_diff)

    lbp_diffs = np.array(lbp_diffs) / np.max(lbp_diffs)
    lab_diffs = np.array(lab_diffs) / np.max(lab_diffs)
    flow_diffs = np.array(flow_diffs) / np.max(flow_diffs)
    edge_diffs = np.array(edge_diffs) / np.max(edge_diffs)

    combined_diffs = np.column_stack((lbp_diffs, lab_diffs, flow_diffs, edge_diffs))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(combined_diffs)
    weights = kmeans.cluster_centers_.mean(axis=0)
    weights /= weights.sum()

    weighted_diffs = np.dot(combined_diffs, weights)

    rolling_mean = np.convolve(weighted_diffs, np.ones(15) / 15, mode='same')
    rolling_std = np.convolve(weighted_diffs, np.ones(15) / 15, mode='same')
    adaptive_threshold = rolling_mean + 0.35 * rolling_std 

    changes, _ = find_peaks(weighted_diffs, height=adaptive_threshold, distance=12) 

    return changes, weighted_diffs


def classify_changes(changes, diffs, window_size=25):
    gradual_changes = []
    abrupt_changes = []

    for change in changes:
        if change < window_size:
            abrupt_changes.append(change)
        else:
            pre_change = diffs[max(0, change - window_size):change]
            post_change = diffs[change:min(len(diffs), change + window_size)]

            if np.mean(post_change) > np.mean(pre_change) * 1.8:  
                abrupt_changes.append(change)
            else:
                if np.max(post_change) > np.mean(pre_change) * 2.5: 
                    abrupt_changes.append(change)
                else:
                    pre_change_std = np.std(pre_change)
                    post_change_std = np.std(post_change)

                    if (np.mean(post_change) < np.mean(pre_change) * 0.5 or  
                            post_change_std > pre_change_std * 1.25):  
                        continue 

                    gradual_changes.append(change)

    return gradual_changes, abrupt_changes


def filter_changes(changes, min_distance=15):
    filtered = [changes[0]]
    for change in changes[1:]:
        if change - filtered[-1] >= min_distance:
            filtered.append(change)
    return filtered


frames_folder = "folder_name"
changes, diffs = detect_changes(frames_folder)
gradual_changes, abrupt_changes = classify_changes(changes, diffs)
gradual_changes = filter_changes(gradual_changes)
abrupt_changes = filter_changes(abrupt_changes)

gradual_changes = [int(frame) for frame in gradual_changes]
abrupt_changes = [int(frame) for frame in abrupt_changes]

print("Gradual changes at frames:", gradual_changes)
print("Abrupt changes at frames:", abrupt_changes)
print(len(gradual_changes), len(abrupt_changes))