import streamlit as st
from streamlit_player import st_player
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

############################################################
############################################################
import os
import collections
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import core.yolov4
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
# initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)
# load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, True)
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
XYSCALE = cfg.YOLO.XYSCALE
FRAMEWORK = 'tf'
input_size = 416
video_path = './data/video/fall_sample2.mp4'
saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
############################################################
############################################################

DEMO_VIDEO = 'demo_video.mp4'

st.title('Fall Detection Application Using YOLO')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Menu')
# st.sidebar.subheader('Parameters')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Please Select',
['About', 'Sample Videos', 'Help', 'Run on Video']
)

if app_mode =='About':
    st.markdown('''
    This is an application for fall detection of individuals based on the **YOLO V.4** object detection algorithm.
    The method used in this algorithm is suitable for detecting falls from a standing position or while walking. \n
    This method is based on the proposed method in **Lu, K. L., & Chu, E. T. H. (2018). 
    An image-based fall detection system for the elderly. Applied Sciences, 8(10), 1995.**
    ''')

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.image('TEAM_LOGO.jpg')

elif app_mode == 'Sample Videos':
    st.video('demo1.mp4', format='video/mp4', start_time=0)
    st.video('demo2.mp4', format='video/mp4', start_time=0)
    st.video('demo3.mp4', format='video/mp4', start_time=0)
    st.video('demo4.mp4', format='video/mp4', start_time=0)

elif app_mode == 'Help':
    st.markdown('''
        - The Ratio Factor is a factor which multiplied by the height of the bounding box of 
        the person at 1.5 seconds before each moment. If the height of the bounding box at each 
        moment is less than the multiplication value, the algorithm will detect a falling-down occurrence. 
        The suggested value is 5.5, but values between 5 and 7 are good choices. The higher values will lead to more 
        conservative results. \n
        ''')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


####################################################################
####################################################################

elif app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown('---')
    ratio = st.sidebar.slider('Ratio', min_value=1.0, max_value=8.0, value=5.5, step=0.5)
    st.sidebar.markdown('---')
    st.markdown(' ## Output')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=['mp4'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_res.avi', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)
    fps = 0
    i = 0

    kpi1, kpi2, kpi3 = st.beta_columns(3)

    with kpi1:
        kpi1 = st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Tracked Individuals**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Fall Detection Status**")
        kpi3_text = st.markdown('')
        kpi3_text.write(f"<h1 style='text-align: center; color: green;'>{'No Fall'}</h1>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    ###################################################
    ###################################################
    frame_num = 0
    # while video is running
    # DEFINING A DICTIONARY FOR TRACKING
    id_Locs = collections.defaultdict(list)  # FOR METHOD THREE
    id_ylocs = collections.defaultdict(list)  # FOR METHOD ONE
    yLocs = []
    falls = 0
    track_dict = dict()
    frame_list = []

    while vid.isOpened():
        i += 1
        ret, frame = vid.read()
        if not ret:
            continue
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.3,
            score_threshold=0.2
        )
        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]
        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]
        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']
        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        # cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
        #             (0, 255, 0), 2)
        # print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]
        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            # cv2.circle(frame, (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)), 5, color, -1)
            # cv2.circle(frame, (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)), 15, (0, 255, 0), -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)
            #################################################
            ##  PAPER METHOD FOR FALL DETECTION #############
            #################################################
            frameRate = 25
            id_Locs[track.track_id].append([int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0])])
            for key, value in id_Locs.items():
                if len(value) > int(np.floor(frameRate * 1.5)):  # 1.5econds after detection a person:
                    # if value[-1][0] < (7/8) * value[-1 * int(np.floor(frameRate * 1.5))][0]:
                    # if value[-1][0] < (5.5 / 8) * value[-1 * int(np.floor(frameRate * 1.5))][0]:
                    if value[-1][0] < (ratio / 8) * value[-1 * int(np.floor(frameRate * 1.5))][0]:
                        print("Fall Detected")
                        cv2.putText(frame, "Person " + str(key) + " Fell Down", (70, 250), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (0, 0, 255), 3)
                        falls += 1
            ########################################################
            # if enable, then print details about each track
            # print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
            #                                                                                     class_name, (
            #                                                                                         int(bbox[0]),
            #                                                                                         int(bbox[1]),
            #                                                                                         int(bbox[2]),
            #                                                                                         int(bbox[3]))))
            each_id_list = [frame_num, str(track.track_id), int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
            frame_list.append(each_id_list)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)

        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{round(fps, 1)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{count}</h1>", unsafe_allow_html=True)

        if falls > 0:
            cv2.putText(frame, "Fall Detected", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 5)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{'Fall Detected'}</h1>", unsafe_allow_html=True)

        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        frame = image_resize(image=frame, width=640)
        stframe.image(frame, channels='RGB', use_column_width=True)
        out.write(frame)

    vid.release()
    out.release()

