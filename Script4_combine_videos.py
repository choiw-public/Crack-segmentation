import glob
import cv2 as cv
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt

raw_video_dir = "./video_to_combine(392600)/raw"
mask_video_dir = "./video_to_combine(392600)/mask"
main_video_dir = "./video_to_combine(392600)/main"
url_file = "./video_to_combine(392600)/video_source"
dst_dir = "./video_to_combine(392600)/combined_x3"

canvas_width = 1920
canvas_height = 1080
edge_thickness = 5
seperator_thickness = 5

canvas_main_frame_width_proportion = 0.7
fps_modifier = 3.0

# ========================================================= #
raw_video_list = sorted(list(glob.glob(os.path.join(raw_video_dir, "*"))))
mask_video_list = sorted(list(glob.glob(os.path.join(mask_video_dir, "*"))))
main_video_list = sorted(list(glob.glob(os.path.join(main_video_dir, "*"))))

title_height = 30  # should me manually defined

canvas_main_frame_width = int(canvas_width * canvas_main_frame_width_proportion)
canvas_side_frame_width = canvas_width - canvas_main_frame_width

# calculate allowable frame size of raw and mask frames
raw_h_limit = int(canvas_height * 0.5 - title_height - edge_thickness * 2)
raw_w_limit = canvas_side_frame_width - edge_thickness * 2
mask_h_limit = raw_h_limit
mask_w_limit = raw_w_limit
main_h_limit = canvas_height - title_height - edge_thickness * 2
main_w_limit = canvas_main_frame_width - seperator_thickness - edge_thickness * 2

os.makedirs(dst_dir, exist_ok=True)

# calculate allowable frame size of main frame

# inspect file name consistency
for name_raw, name_mask, name_main in zip(raw_video_list, mask_video_list, main_video_list):
    if not os.path.basename(name_raw)[:-4] == os.path.basename(name_mask)[:-4] == os.path.basename(name_main)[:-4]:
        raise ValueError("file names are not consistent")


def draw_text(img, text, left_top_coord, size):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text(left_top_coord, text, font=ImageFont.truetype("FreeMonoBold.ttf", size))
    return np.array(img)


def make_edge(img):
    h, w, _ = img.shape
    vertical = np.ones((h, edge_thickness)).astype(np.uint8) * 128
    vertical = np.stack([np.zeros_like(vertical), vertical, np.zeros_like(vertical)], 2)
    img = np.concatenate([vertical, img, vertical], 1)
    horizontal = np.ones((edge_thickness, w + edge_thickness * 2)).astype(np.uint8) * 128
    horizontal = np.stack([np.zeros_like(horizontal), horizontal, np.zeros_like(horizontal)], 2)
    img = np.concatenate([horizontal, img, horizontal], 0)
    return img


def combine(raw_frame, mask_frame, main_frame, source):
    canvas = np.zeros((canvas_height, canvas_width, 3)).astype(np.uint8)

    # resizing - raw frame
    frame_height, frame_width, _ = raw_frame.shape
    raw_frame_resize_factor = min(float(raw_h_limit) / float(frame_height), float(raw_w_limit) / float(frame_width))
    raw_frame = cv.resize(raw_frame, (int(frame_width * raw_frame_resize_factor), int(frame_height * raw_frame_resize_factor)))
    raw_frame = make_edge(raw_frame)
    raw_frame_h, raw_frame_w, _ = raw_frame.shape

    # resizing - mask frame
    mask_frame = cv.resize(mask_frame, (int(frame_width * raw_frame_resize_factor), int(frame_height * raw_frame_resize_factor)))
    mask_frame = make_edge(mask_frame)
    mask_frame_h, mask_frame_w, _ = mask_frame.shape

    # resizing - main frame
    main_frame_resize_factor = min(float(main_h_limit) / float(frame_height), float(main_w_limit) / float(frame_width))
    main_frame = cv.resize(main_frame, (int(frame_width * main_frame_resize_factor), int(frame_height * main_frame_resize_factor)))
    main_frame = make_edge(main_frame)
    main_frame_h, main_frame_w, _ = main_frame.shape

    # coordinate raw and mask frames
    raw_top_left_y = int((canvas_height * 0.5 - raw_frame_h - title_height) * 0.5) + title_height
    raw_top_left_x = int((canvas_side_frame_width - raw_frame_w) * 0.5)
    mask_top_left_y = int(canvas_height * 0.5) + int((canvas_height * 0.5 - mask_frame_h - title_height) * 0.5) + title_height

    canvas = draw_text(canvas, "Input video", (raw_top_left_x + edge_thickness, raw_top_left_y - title_height), 25)
    canvas[raw_top_left_y:raw_top_left_y + raw_frame_h, raw_top_left_x:raw_top_left_x + raw_frame_w, :] = raw_frame
    canvas = draw_text(canvas, "Prediction", (raw_top_left_x + edge_thickness, mask_top_left_y - title_height), 25)
    canvas[mask_top_left_y:mask_top_left_y + mask_frame_h, raw_top_left_x:raw_top_left_x + mask_frame_w, :] = mask_frame

    # handling main frame
    main_top_left_y = int((canvas_height - main_frame_h - title_height) * 0.5) + title_height
    main_top_left_x = int((canvas_width - (canvas_side_frame_width + seperator_thickness + main_frame_w)) * 0.5) + canvas_side_frame_width + seperator_thickness
    canvas = draw_text(canvas, "Result", (main_top_left_x + edge_thickness, main_top_left_y - title_height), 25)
    canvas[main_top_left_y:main_top_left_y + main_frame_h, main_top_left_x:main_top_left_x + main_frame_w, :] = main_frame
    canvas = draw_text(canvas, source, (raw_top_left_x + edge_thickness, raw_top_left_y + edge_thickness), 20)
    canvas = draw_text(canvas, "From a customized deep learning model", (raw_top_left_x + edge_thickness, mask_top_left_y + edge_thickness), 20)
    # canvas = draw_text(canvas, "Superimposed prediction on input video", (main_top_left_x + edge_thickness, main_top_left_y + edge_thickness), 20)
    return canvas


with open(url_file, "r") as reader:
    url_list_tmp = reader.readlines()

url_list = dict()
for url in url_list_tmp:
    url = url.split(", ")
    url_list[url[0]] = url[1].replace("\n", "")

for frame_number, (name_raw, name_mask, name_main) in enumerate(zip(raw_video_list, mask_video_list, main_video_list)):
    raw_video = cv.VideoCapture(name_raw)
    mask_video = cv.VideoCapture(name_mask)
    main_video = cv.VideoCapture(name_main)
    raw_frame_exist, frame_raw = raw_video.read()
    mask_frame_exist, frame_mask = mask_video.read()
    main_frame_exist, frame_main = main_video.read()

    if not raw_frame_exist or not mask_frame_exist or not main_frame_exist:
        raise ValueError("no frame exists")
    else:
        num_frames = 0
        should_continue = True
        fps_raw = round(raw_video.get(5))
        fps_mask = round(mask_video.get(5))
        fps_main = round(main_video.get(5))
        is_same_fps = fps_raw == fps_mask == fps_main
        if not is_same_fps:
            raise ValueError("differne fps")
        num_frames_raw = raw_video.get(7)
        num_frames_mask = mask_video.get(7)
        num_frames_main = main_video.get(7)

        is_same_num_frames = num_frames_raw == num_frames_mask == num_frames_main

        if not is_same_num_frames:
            raise ValueError("different number of frames")

        output_fps = fps_raw * fps_modifier
        base_name_without_ext = os.path.basename(name_raw)[:-4]
        source_url = url_list[base_name_without_ext]
        combined_video_name = os.path.join(dst_dir, base_name_without_ext + ".avi")
        combined_video = cv.VideoWriter(combined_video_name, cv.VideoWriter_fourcc(*"XVID"), output_fps, (canvas_width, canvas_height))
        combined_frame = combine(frame_raw, frame_mask, frame_main, source_url)
        combined_video.write(combined_frame)
        num_frames += 1
        print("%s (%05d/%05d)" % (base_name_without_ext, num_frames, num_frames_raw))
        while should_continue:
            raw_frame_exist, frame_raw = raw_video.read()
            mask_frame_exist, frame_mask = mask_video.read()
            main_frame_exist, frame_main = main_video.read()
            if raw_frame_exist or mask_frame_exist or main_frame_exist:
                should_continue = True
                combined_frame = combine(frame_raw, frame_mask, frame_main, source_url)
                combined_video.write(combined_frame)
                num_frames += 1
                if num_frames % 10 == 0:
                    print("%s (%05d/%05d)" % (base_name_without_ext, num_frames, num_frames_raw))
            else:
                should_continue = False
