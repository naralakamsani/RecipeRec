import os
import sys
import numpy as np
import logging as log
from openvino.inference_engine import IECore
import tensorflow as tf
from datetime import datetime
import cv2

def infer():
    start=datetime.now()

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    cpu_extension = None
    device = 'CPU'
    input_files = ['uploads/' + f for f in os.listdir('Uploads') if os.path.isfile(os.path.join('Uploads', f))]
    number_top = 1
    labels = 'labels.txt'
    predicted_ingredients = set()

    if labels:
        with open(labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = [None]

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, "CPU")

    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    model = "models/openvino/MobileNetV2/mobilenetv2.xml"
    weights = "models/openvino/MobileNetV2/mobilenetv2.bin"
    tflite_model = "models/TFLITE/lite-model_object_detection_mobile_object_localizer_v1_1_metadata_2.tflite"
    log.info(f"Loading network:\n\t{model}")

    net = ie.read_network(model=model, weights=weights)

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    # Initialize TFLITE interpreter
    interpreter = tf.lite.Interpreter(model_path = tflite_model)
    interpreter.allocate_tensors()

    #print model metadata
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device)

    # Open Video Capture
    cap = cv2.VideoCapture(input_files[0])
    if cap.isOpened() == False:
        print("Error opening video file")
    print(input_files[0].split('.')[-1])
    if input_files[0].split('.')[-1] in ['jpeg', 'jpg', 'png']:
        log.info("Starting inference in synchronous mode")
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            frame = np.transpose(frame, [2,0,1]) / 255
            frame = np.reshape(frame, (1, 3, 224, 224))
            res = exec_net.infer(inputs={input_blob: frame})

            # Processing output blob
            log.info("Processing output blob")
            res = res[out_blob]

            log.info("results: ")
            for i, probs in enumerate(res):
                probs = np.squeeze(probs) #[np.squeeze(probs) > .5]
                top_ind = np.argsort(probs)[-number_top:][::-1]

                for id in top_ind:
                    det_label = labels_map[id] if labels_map else "{}".format(id)
                    predicted_ingredients.add(det_label)
                    print(det_label)
    else:
        fcount = 0
        while cap.isOpened():
            # Capture each frame - slower than native openvino inference
            ret, frame = cap.read()
            fcount += 1
            if (ret == True) and ((fcount % 10) == 0):
                print(f'frame: {fcount}')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cropped = frame[:, 420:1500] # 1080x1080
                small = cv2.resize(cropped, (192,192), interpolation = cv2.INTER_AREA)
                small = np.resize(small, (1, 192, 192, 3))
                interpreter.set_tensor(input_details[0]['index'], small)
                interpreter.invoke()
                out_dict = {
                    'detection_boxes' : interpreter.get_tensor(output_details[0]['index']),
                    'detection_scores' : interpreter.get_tensor(output_details[2]['index'])}
                out_dict['detection_boxes'] = out_dict['detection_boxes'][0][:number_top]
                out_dict['detection_scores'] = out_dict['detection_scores'][0][:number_top]
                for i, score in enumerate(out_dict['detection_scores']):
                    if score > .5:
                        ymin, xmin, ymax, xmax = (out_dict['detection_boxes'][i]*1080).astype(int)
                        # print((ymin, xmin, ymax, xmax))
                        roi = cropped[ymin:ymax, xmin:xmax]
                        if roi.shape[0] < 80 or roi.shape[1] < 80:
                            continue
                        roi = cv2.resize(roi, (224, 224), interpolation = cv2.INTER_AREA)
                        #cv2.imshow('test', cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                        #cv2.waitKey(250)
                        #cv2.destroyAllWindows()
                        roi = np.transpose(roi, [2,0,1]) / 255
                        roi = np.reshape(roi, (1, 3, 224, 224))
                        # Start sync inference
                        res = exec_net.infer(inputs={input_blob: roi})

                        # Processing output blob
                        res = res[out_blob]

                        for i, probs in enumerate(res):
                            probs = np.squeeze(probs) #[np.squeeze(probs) > .5]
                            top_ind = np.argsort(probs)[-number_top:][::-1]
                            if probs[top_ind] < .7:
                                continue
                            for id in top_ind:
                                det_label = labels_map[id] if labels_map else "{}".format(id)
                                predicted_ingredients.add(det_label)
                                print(det_label)
            elif (ret == False):
                cap.release()

    print()
    print("Time for program to run is:")
    print(datetime.now()-start)
    print()
    print("Predicted ingredients:")
    print(predicted_ingredients)

    return predicted_ingredients