#!/usr/bin/env python3

import sys
import numpy as np

import pathlib
temp = pathlib.WindowsPath
pathlib.WindowsPath = pathlib.PosixPath

import argparse
import torch
import cv2
import pyzed.sl as sl
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from threading import Lock, Thread
from time import sleep
import keyboard

import cv_viewer.tracking_viewer as cv_viewer

import arduino_communication as ac
import database_communication as dc
import time
import pandas as pd
from utility import convert_sensor_data_to_dataframe
from datetime import datetime

# For boolean flags, writing to a boolean variable is an atomic operation, which means it cannot be interrupted by a context switch to another thread.
lock = Lock()
run_signal = False
exit_signal = False


def img_preprocess(img, device, half, net_size):
    net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
    net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    net_image = np.ascontiguousarray(net_image)

    img = torch.from_numpy(net_image).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, ratio, pad


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output


def detections_to_custom_box(detections, im, im0):
    output = []
    for i, det in enumerate(detections):
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                # Creating ingestable objects for the ZED SDK
                obj = sl.CustomBoxObjectData()
                obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
                obj.label = cls
                obj.probability = conf
                obj.is_grounded = False
                output.append(obj)
    return output


def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")

    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = img_size

    # Load model
    model = attempt_load(weights, device=device)  # load FP32
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    cudnn.benchmark = True

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    print("Network Initialized. Running Torch Thread")
    while not exit_signal:
        if run_signal:
            lock.acquire()
            img, ratio, pad = img_preprocess(image_net, device, half, imgsz)

            pred = model(img)[0]
            det = non_max_suppression(pred, conf_thres, iou_thres)

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, img, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)
    run_signal = False
    exit_signal = True
    print("Exiting Torch Thread")

def sensor_measurements_thread(logging):
    global conn,exit_signal
    REFRESH_RATE = 5
    NUM_NODES = 2
    # Initialize the serial communication
    ser = ac.initialize_communication()
    # Check if the serial connection is open
    if ser.is_open:
        print("Serial connection established.")
    else:
        print("Failed to establish serial connection.")
        exit_signal = True


    # Make sure Arduino Communication is stable
    if not exit_signal:
        for i in range(3):
            ac.receive_arduino_communication(ser)

        if ac.receive_arduino_communication(ser) == {}:
            print("Failed to receive data from the Arduino.")
            print("Restart the program")
            ac.close_communication(ser)
            exit_signal = True
    #Logging
    log_df = pd.DataFrame()
    
    # Main Loop
    
    while not exit_signal:
        # Receive data from the Arduino
        arduino_data = ac.receive_arduino_communication(ser)
        if arduino_data:
            node_measurements = convert_sensor_data_to_dataframe(arduino_data, NUM_NODES)
            lock.acquire()
            dc.upsert_node_measurements(conn, node_measurements)
            lock.release()
            if logging:
                log_df =pd.concat([log_df,node_measurements],ignore_index=True)
        
        time.sleep(REFRESH_RATE)
    
    # Save the log
    if logging and not log_df.empty:
        datetime_string = datetime.now().strftime("%d%m%Y_%H%M%S")
        log_name = "logs/log_"+datetime_string+".csv"
        log_df.to_csv(log_name, index=False)

    # Close all communications
    if ser.is_open:
        ac.close_communication(ser)
    if conn:
        dc.close_conn(conn)
    exit_signal = True
    print("Exiting Sensor Thread")
    
def check_for_exit():
    global exit_signal
    while True:
        if keyboard.is_pressed('esc'):
            exit_signal = True
            break
        sleep(0.1)  # To prevent CPU overuse

def main():
    global image_net, exit_signal, run_signal, detections, conn

    conn = dc.initialize_conn()

    """ Start Threads"""
    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    sensors_thread = Thread(target=sensor_measurements_thread,kwargs={'logging':opt.log})
    sensors_thread.start()

    Thread(target=check_for_exit).start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    image_left = sl.Mat()
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()

    print("To exit press 'esc'")
    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal and not exit_signal:
                sleep(0.001)

            # -- In case another thread finishes
            if exit_signal:
                break
            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            if opt.viewer:
                cv2.imshow("ZED | 2D View and Birds View", global_image)
            
                # To exit press 'esc'
                key = cv2.waitKey(10)
                if key == 27:
                    exit_signal = True
                
            # for object in objects.object_list:
            #     print("ID: {}, Pos: {}, Vel: {}".format(object.id, object.position,object.velocity))
        else:
            exit_signal = True
    exit_signal = True
    zed.close()
    print("Camera closed.")
    print("Exiting Main Thread")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best_weights.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--viewer', action='store_true', help='Display viewer for debugging purposes')
    parser.add_argument('--log', action='store_true', help='Log the sensor measurements')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
