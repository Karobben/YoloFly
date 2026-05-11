# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os, math
import sys
import queue
import threading
from pathlib import Path
import pandas as pd
import cv2, json
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, strip_optimizer, xyxy2xywh, LOGGER
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


def Frame_monochrom(frame):
    frame = cv2.resize(frame, (120,90))
    frame_grey = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    (thresh, frame) = cv2.threshold(frame_grey, 50, 255, cv2.THRESH_BINARY)
    frame[frame==255]=1
    frame = np.array(frame, dtype="uint16")
    return frame


def _prefetch_video_frames(dataset):
    """Yield (path, img, im0s, vid_cap, s) with next frame decoded in a background thread (overlaps decode with inference)."""
    q = queue.Queue(maxsize=1)

    def producer():
        try:
            for item in dataset:
                q.put(item)
        except Exception:
            q.put(None)
            raise
        q.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item


def _load_yolo_rows_for_tracking_init(label_path: str):
    """Read YOLO txt rows for initializing tracking on the first tracked frame."""
    p = Path(str(label_path)).expanduser()
    if not p.is_file():
        return None
    rows = []
    try:
        lines = p.read_text(encoding='utf-8').splitlines()
    except OSError:
        return None
    for raw in lines:
        t = str(raw).strip()
        if not t:
            continue
        cols = t.split()
        if len(cols) < 5:
            continue
        try:
            nums = [float(x) for x in cols[:6]]
        except ValueError:
            continue
        # Accept either cls+xywh (5 cols) or cls+xywh+conf (>=6 cols).
        if len(cols) < 6:
            nums = nums[:5] + [1.0]
        else:
            nums = nums[:6]
        nums[0] = int(round(nums[0]))
        rows.append(" ".join(str(v) for v in nums))
    return rows if rows else None


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        img_save = False,
        img_save_frame = False,
        bh_count = False,
        chain_det = False,
        tar_track = False,
        tar_tr_start = 1,
        tracks_show = False,
        path_ink = False,
        head_bind = False,
        num_fly = '0',
        debug = False,
        quiet = False,
        snap_frame=None,  # None: full run; int (1-based): single video frame, save annotated PNG and stop
        frame_skip=1,  # process one frame every N frames (quick screening)
        frame_start=None,  # None: start from first frame; int (1-based): inclusive start
        frame_end=None,  # None: process to end; int (1-based): inclusive end
        tracking_dir='csv',  # where detect CSV + tracking JSON are written
        init_label_path=None,  # optional local YOLO txt used for first tracked frame bootstrap
        ):
    source = str(source)
    frame_skip = max(1, int(frame_skip or 1))
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if dnn:
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        if pt and device.type != 'cpu':
            cudnn.benchmark = True  # speed up fixed-size video inference
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Optional frame window (single-video file only): process inclusive [frame_start, frame_end].
    if frame_start is not None:
        frame_start = int(frame_start)
        if frame_start < 1:
            raise ValueError('--frame-start must be >= 1.')
    if frame_end is not None:
        frame_end = int(frame_end)
        if frame_end < 1:
            raise ValueError('--frame-end must be >= 1.')
    if frame_start is not None and frame_end is not None and frame_end < frame_start:
        raise ValueError('--frame-end must be >= --frame-start.')
    if snap_frame is not None and (frame_start is not None or frame_end is not None):
        raise ValueError('--snapshot-frame cannot be combined with --frame-start/--frame-end.')

    frame_window_active = frame_start is not None or frame_end is not None
    window_start = 1 if frame_start is None else int(frame_start)
    window_end = None if frame_end is None else int(frame_end)

    if frame_window_active:
        if webcam:
            raise ValueError('--frame-start/--frame-end are only supported for file video sources, not streams/webcam.')
        if dataset.nf != 1 or not getattr(dataset, 'video_flag', [False])[0]:
            raise ValueError('--frame-start/--frame-end require a single video file as --source.')
        n_frames = int(getattr(dataset, 'frames', 0) or 0)
        if n_frames <= 0:
            raise ValueError('Could not read frame count for video; --frame-start/--frame-end not supported.')
        if window_start > n_frames:
            raise ValueError(f'--frame-start {window_start} out of range 1..{n_frames} for this video.')
        if window_end is None:
            window_end = n_frames
        elif window_end > n_frames:
            raise ValueError(f'--frame-end {window_end} out of range 1..{n_frames} for this video.')
        if dataset.cap is not None and window_start > 1:
            dataset.cap.set(cv2.CAP_PROP_POS_FRAMES, window_start - 1)

    # Single-frame snapshot: one 1-based video frame index (matches Num_frame in this script)
    if snap_frame is not None:
        if webcam:
            raise ValueError('--snapshot-frame is only supported for file/dir video or image sources, not streams/webcam.')
        if dataset.nf != 1 or not getattr(dataset, 'video_flag', [False])[0]:
            raise ValueError('--snapshot-frame requires a single video file as --source.')
        n_frames = int(getattr(dataset, 'frames', 0) or 0)
        if n_frames <= 0:
            raise ValueError('Could not read frame count for video; --snapshot-frame not supported.')
        if snap_frame < 1 or snap_frame > n_frames:
            raise ValueError(f'--snapshot-frame {snap_frame} out of range 1..{n_frames} for this video.')
        if snap_frame > 1 and dataset.cap is not None:
            dataset.cap.set(cv2.CAP_PROP_POS_FRAMES, snap_frame - 1)

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    Num_frame = (window_start - 1) if frame_window_active else 0

    ############
    ### Karobben
    ############
    if path_ink:
        from utils import PathInk
        frame_tmp = "NA"

    if chain_det:
        from utils import Chain_detect

    if tar_track:
        from utils import Fly_Tra
    if head_bind:
        from utils.Head_bind import head_match

    if img_save_frame:
        Save_lsit = open("list").read().split("\n")[:-1]
        Save_lsit = [int(i) for i in Save_lsit]

    # Karobben: behaviors count
    Video = source.split("/")[-1]
    csv_dir = Path(str(tracking_dir) if tracking_dir is not None else "csv")
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_name = Video + (f"_Fskip_{frame_skip}" if frame_skip > 1 else "") + ".csv"
    csv_path = csv_dir / csv_name
    if csv_path.exists():
        csv_path.unlink(missing_ok=True)
    csv_buffer = []  # buffer and write once at end (faster)
    # Karobben: Tracking save
    if tar_track:
        json_path = csv_dir / (Video + "_" + str(tar_tr_start) + "_.json")
        if json_path.exists():
            json_path.unlink(missing_ok=True)
        Trac_out = open(json_path, "a")
        json_buffer = []  # buffer and write once at end (faster)

    init_label_rows = None
    if tar_track and init_label_path:
        init_label_rows = _load_yolo_rows_for_tracking_init(str(init_label_path))
        if init_label_rows is None:
            LOGGER.warning(f'Could not load --init-label-path, fallback to model output: {init_label_path}')
        else:
            LOGGER.info(f'Loaded {len(init_label_rows)} rows from init label: {init_label_path}')

    # Progress bar and prefetch for single-video processing (frame-based)
    pbar = None
    use_prefetch = not webcam and dataset.nf == 1 and getattr(dataset, 'video_flag', [False])[0]
    if use_prefetch:
        total_frames = getattr(dataset, 'frames', 0) or 0
        if frame_window_active and window_end is not None:
            total_frames = max(0, int(window_end) - int(window_start) + 1)
        if total_frames > 0:
            pbar = tqdm(total=total_frames, unit='frame', desc='Video', dynamic_ncols=True)
    frame_iter = _prefetch_video_frames(dataset) if use_prefetch else dataset
    head_match_instance = None  # reuse across frames to avoid per-frame alloc
    gn_cache, gn_cache_shape = None, None  # cache normalization gain when im0 shape unchanged (video)

    for path, img, im0s, vid_cap, s in frame_iter:
        if pbar is not None:
            pbar.update(1)
        if snap_frame is not None:
            Num_frame = snap_frame  # 1-based index for this decode (after optional cap seek)
        else:
            Num_frame += 1  ## Karobben - -
        if frame_window_active:
            if Num_frame < window_start:
                continue
            if window_end is not None and Num_frame > window_end:
                break
        if snap_frame is None and frame_skip > 1 and (Num_frame - 1) % frame_skip != 0:
            continue
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = torch.tensor(net.forward())
            else:
                pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        snapshot_done = False
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            shp = (im0.shape[0], im0.shape[1])
            if gn_cache is None or gn_cache_shape != shp:
                gn_cache = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                gn_cache_shape = shp
            gn = gn_cache
            #print("shape", im0.shape, frame)
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            Lable_Result = []

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                #print("\n\nLet's test\n\n")
                #print(reversed(det))
                for *xyxy, conf, cls in reversed(det):
                    #if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # Always keep confidence in label rows saved to csv outputs.
                    cls_id = int(round(cls.item()))
                    label_tmp = [str(cls_id)] + [str(i) for i in xywh] + [str(float(conf.item()))]
                    Lable_Result.append(" ".join(label_tmp))
                    #print("line is here", label_tmp)
                    #with open(txt_path + '.txt', 'a') as f:
                    #    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        if cls >= 0:
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            if not quiet:
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()


            ############
            ### Karobben
            ############
            if chain_det:
                '''
                Detect the chains by distance
                '''
                TB = pd.DataFrame([i.split(" ") for i in Lable_Result]).astype(float)
                TB[0] = pd.to_numeric(TB[0])
                AA = Chain_detect.Chain_finder(TB)
                #annotator.box_label((100, 500,200,700), "I am Here!!", color=(0,0,0))
                if view_img:
                    for Box in AA.Chain_result:
                        BOX = ( (Box[0][0]-Box[0][2]/2) * len(im0[0]),
                                (Box[0][1]-Box[0][3]/2) * len(im0),
                                (Box[1][0]+Box[1][2]/2) * len(im0[0]),
                                (Box[1][1]+Box[1][3]/2) * len(im0)
                                )
                        annotator.box_label(BOX, "Chain_chasing", color=(0,0,0))
                if bh_count:
                    for Box in AA.Chain_result:
                        POS_re = ["3.5", (Box[0][0]+Box[0][2]/2),
                                (Box[0][1]+Box[0][3]/2),
                                (abs(Box[1][0]-Box[1][2])/2),
                                (abs(Box[1][1]-Box[1][3])/2),
                                "1.0"
                                ]
                        POS_re = [str(i) for i in POS_re]
                        Lable_Result.append(" ".join(POS_re))

            if tar_track and Num_frame >= tar_tr_start:
                tracking_rows = Lable_Result
                if Num_frame == tar_tr_start and init_label_rows is not None:
                    tracking_rows = list(init_label_rows)
                    # Keep downstream CSV/json outputs aligned with the initialized rows.
                    Lable_Result = list(init_label_rows)
                TB = pd.DataFrame([i.split(" ") for i in tracking_rows]).astype(float)
                TB_head = TB[TB[0]==1]
                TB_head.index = range(len(TB_head.index))
                #print(TB)
                if Num_frame == tar_tr_start:
                    fly_align = Fly_Tra.fly_align()

                    def FLY_matrix_update(FLY_matrix, FLY_matrix_tmp, frame):
                        MATCH_result = fly_align.align(FLY_matrix, FLY_matrix_tmp, frame)
                        FLY_LIST = {v: k for k, v in MATCH_result.items()}
                        FLY_LIST = {v: k for k, v in sorted(FLY_LIST.items(), key=lambda item: item[1])}
                        Dic_update = {frame: {}}
                        for i in FLY_LIST:
                            Dic_update[frame].update({i: FLY_matrix_tmp[frame][FLY_LIST[i]]})
                        FLY_matrix.update(Dic_update)
                        LOST_ID = list(FLY_matrix[Num_frame - 1].keys())
                        fly_new = list(FLY_matrix[Num_frame].keys())
                        [LOST_ID.remove(i) for i in fly_new]
                        if len(LOST_ID) > 0:
                            for id_old in LOST_ID:
                                FLY_matrix[Num_frame].update({id_old: FLY_matrix[Num_frame - 1][id_old]})
                        return FLY_matrix

                if Num_frame == tar_tr_start:
                    FLY_matrix = fly_align.TB_dic(TB, Num_frame)

                    if num_fly == 0:
                        print("Them number of the fly was determined by the frist positive result of the frame")
                    else:
                        print("The number of expected fly is:", num_fly)

                    if head_bind:
                        # try if there are enogh head with match the flies. Or we skip to next frame
                        try:
                            if len(FLY_matrix[Num_frame]) != num_fly:
                                print(I_am_a_error_for_release_the_try)
                            if head_match_instance is None:
                                head_match_instance = head_match()
                            head_bind = head_match_instance
                            head_bind.main(FLY_matrix, Num_frame, TB_head)
                            if debug:
                                print("Head Match:", head_bind.MATCH_result)
                            else:
                                pass
                            for fly in FLY_matrix[Num_frame].keys():
                                #print(int(head_bind.MATCH_result[fly]))
                                #print(TB_head)
                                #print(TB_head.iloc[int(head_bind.MATCH_result[fly]),1:])
                                FLY_matrix[Num_frame][fly].update({"head":list(TB_head.iloc[int(head_bind.MATCH_result[fly]),1:])})                        #print(FLY_matrix)
                            dic_ID = list(FLY_matrix.keys())[-1]
                            tmp = {dic_ID:FLY_matrix[dic_ID]}
                            FLY_matrix_tmp_str = json.dumps(tmp) +";"
                            if debug:
                                print(FLY_matrix_tmp_str)
                            else:
                                pass
                            json_buffer.append(FLY_matrix_tmp_str)
                        except:
                            Trac_out.write(''.join(json_buffer))
                            Trac_out.close()
                            json_buffer = []
                            tar_tr_start += 1
                            json_path = csv_dir / (Video + "_" + str(tar_tr_start) + "_.json")
                            if json_path.exists():
                                json_path.unlink(missing_ok=True)
                            Trac_out = open(json_path, "a")
                            if debug:
                                print("head bind fail, went to next frame:", Num_frame)
                            else:
                                pass
                    else:
                        dic_ID = list(FLY_matrix.keys())[-1]
                        tmp = {dic_ID:FLY_matrix[dic_ID]}
                        FLY_matrix_tmp_str = json.dumps(tmp) +";"
                        #if debug mode
                        if debug:
                            print(FLY_matrix_tmp_str)
                        else:
                            pass
                        json_buffer.append(FLY_matrix_tmp_str)

                if Num_frame > tar_tr_start:
                    # remove previous frame to release memeory
                    if len(FLY_matrix) >= 45:
                        ID = list(FLY_matrix.keys())[0]
                        FLY_matrix.pop(ID)

                    #print("\n\nNearst mach for frame:", Num_frame,"\n\n")
                    FLY_matrix_tmp = fly_align.TB_dic(TB, Num_frame)
                    FLY_matrix = FLY_matrix_update(FLY_matrix, FLY_matrix_tmp, Num_frame)
                    if head_bind:
                        if head_match_instance is None:
                            head_match_instance = head_match()
                        head_bind = head_match_instance
                        head_bind.main(FLY_matrix, Num_frame, TB_head)

                        for fly in FLY_matrix[Num_frame].keys():
                            try:
                                FLY_matrix[Num_frame][fly].update({"head":list(TB_head.iloc[int(head_bind.MATCH_result[fly]),1:])})
                            except:
                                # Inherate the head from previous frame based on relative position
                                last_body = FLY_matrix[Num_frame-1][fly]['body']
                                last_head = FLY_matrix[Num_frame-1][fly]['head']
                                new_body  = FLY_matrix[Num_frame][fly]['body']
                                rel_pos = [last_head[0] - last_body[0], last_head[1] - last_body[1]]
                                rel_pos_new = [rel_pos[0]+ new_body[0], rel_pos[1]+ new_body[1]]
                                FLY_matrix[Num_frame][fly].update({"head": rel_pos_new + last_head[2:4]})
                    # save the FLY_matrix:
                    dic_ID = list(FLY_matrix.keys())[-1]
                    tmp = {dic_ID:FLY_matrix[dic_ID]}
                    FLY_matrix_tmp_str = json.dumps(tmp) +";"
                    #print("Frame in chach",len(FLY_matrix))
                    json_buffer.append(FLY_matrix_tmp_str)

                    FLY_list = [i for i in list(FLY_matrix[Num_frame].keys())]
                    cv2.putText(im0, str(len(FLY_list)) ,(100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 200), 1)
                    cv2.putText(im0, str(round(Num_frame)) ,(100, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100, 200, 200), 1)
                    for flys_id in FLY_list:
                        tc_label_x =  int(FLY_matrix[Num_frame][flys_id]['body'][0] * len(im0[0]))
                        tc_label_y =  int(FLY_matrix[Num_frame][flys_id]['body'][1] * len(im0))
                        cv2.putText(im0, flys_id ,(tc_label_x, tc_label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (102, 205, 0), 5)
                    if tracks_show:
                        def f(n,x):
                            '''
                            Convert numbers
                            '''
                            a=[0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
                            b=[]
                            while True:
                                s=n//x
                                y=n%x
                                b=b+[y]
                                if s==0:
                                    break
                                n=s
                            b.reverse()
                            Result = [a[i] for i in b]
                            return  Result
                        if Num_frame-1 - tar_tr_start <= 45:
                            frame_start = tar_tr_start
                        else:
                            frame_start = Num_frame - 1 -45
                        for frame_track in range(frame_start, Num_frame-1):
                            FLY_id = 0
                            FLY_list = [flys_id for flys_id in list(FLY_matrix[Num_frame].keys())]
                            FLY_list.sort()
                            #print(FLY_list)
                            for flys_id in FLY_list:
                                POSITION = (int((FLY_matrix[frame_track+1][flys_id]['body'][0] * len(im0[0]))),                       int(FLY_matrix[frame_track+1][flys_id]['body'][1] * len(im0)))
                                track_Color = [255, 255, 255]
                                for i_num in f(FLY_id,3):
                                    track_Color[i_num]=track_Color[i_num] -50
                                cv2.circle(im0, POSITION, 2, track_Color, thickness=4, lineType=8, shift=0)
                                FLY_id +=1
                                if head_bind:
                                    POSITION_p = (int(FLY_matrix[Num_frame][flys_id]['body'][0] * len(im0[0])),                       int(FLY_matrix[Num_frame][flys_id]['body'][1] * len(im0)))
                                    HEAD_p = (int(FLY_matrix[Num_frame][flys_id]['head'][0] * len(im0[0])),                       int(FLY_matrix[Num_frame][flys_id]['head'][1] * len(im0)))
                                    #print(POSITION, HEAD)
                                    cv2.arrowedLine(im0, POSITION_p, HEAD_p, (track_Color), 3, tipLength = 0.5)


                '''
                111
                '''

            if img_save:
                similar_thread = 20000
                try:
                    img_now = Frame_monochrom(im0)
                    img_similar = sum(abs(img_tmp-img_now)).sum()
                    #print("\n\nWhat a bs\n\n", img_similar)
                    if img_similar > similar_thread:
                        img_tmp = img_now
                except:
                    img_tmp = Frame_monochrom(im0)
                    img_similar = 1 + similar_thread
                #print("img_similar", img_similar)
                #if Num_frame:# % 30 ==0:
                #if Num_frame % 30 ==0:
                if img_similar > similar_thread:
                    cv2.imwrite("../png_DB/png/fly_"+ Video.replace(".","_") + "_frame_"+ str(Num_frame) +'_.png',im0s)
                    F = open("../png_DB/png/fly_"+ Video.replace(".","_") + "_frame_"+ str(Num_frame) +'_.txt', "w")
                    F.write("\n".join(Lable_Result))
                    F.close()

            if img_save_frame:
                if Num_frame in Save_lsit:
                #if img_similar > similar_thread:
                    cv2.imwrite("../png_DB/png/fly_"+ Video.replace(".","_") + "_frame_"+ str(Num_frame) +'_.png',im0s)
                    F = open("../png_DB/png/fly_"+ Video.replace(".","_") + "_frame_"+ str(Num_frame) +'_.txt', "w")
                    F.write("\n".join(Lable_Result))
                    F.close()
                if Num_frame > Save_lsit[-1]:
                    break

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

            Lable_Result_2 = [str(Num_frame)+" "+i for i in Lable_Result]
            if Lable_Result_2:
                csv_buffer.append("\n".join(Lable_Result_2) + "\n")

            if path_ink:
                if frame_tmp == "NA":
                    frame_tmp = im0s
                    frame_tmp = np.array(frame_tmp, dtype="uint16")
                    INK_Result = PathInk.Ink.Frame_monochrom(1, frame_tmp)
                    INK_Result[INK_Result==0] = 0
                    INK_Result[INK_Result==255] = 0
                    #INK_Result = np.array(INK_Result, dtype="uint16")
                    INK_mv_index = []

                INK_all = PathInk.Ink(frame_tmp, im0s, Num_frame, INK_Result, INK_mv_index)
                INK_Result, INK_mv_index, frame_tmp = INK_all.Result, INK_all.mv_index, INK_all.frame_tmp
                #print(INK_Result,"\n\n", INK_mv_index,"\n\n", frame_tmp)

            if snap_frame is not None:
                snap_path = save_dir / f'{p.stem}_frame{snap_frame}_snapshot.png'
                cv2.imwrite(str(snap_path), im0)
                LOGGER.info(f'Snapshot saved to {snap_path}')
                # Also save the raw frame without any drawn labels/boxes.
                snap_raw_path = save_dir / f'{p.stem}_frame{snap_frame}_snapshot_raw.png'
                cv2.imwrite(str(snap_raw_path), im0s)
                LOGGER.info(f'Raw snapshot saved to {snap_raw_path}')
                labels_dir = save_dir / 'labels'
                labels_dir.mkdir(parents=True, exist_ok=True)
                txt_snap = labels_dir / f'{p.stem}_frame{snap_frame}_snapshot.txt'
                with open(txt_snap, 'w') as f:
                    f.write('\n'.join(Lable_Result))
                    if Lable_Result:
                        f.write('\n')
                LOGGER.info(f'Snapshot YOLO labels ({len(Lable_Result)} lines) saved to {txt_snap}')
                snapshot_done = True

        if snap_frame is not None and snapshot_done:
            break

    if pbar is not None:
        pbar.close()
    # karobben result:
    if path_ink:
        ink_TB = pd.DataFrame(INK_Result)
        ink_TB.to_csv( "csv/" + Video + "_ink.csv")
        ink_TB2 = pd.DataFrame(INK_mv_index)
        ink_TB2.to_csv("csv/" + Video + "_mv_index.csv")
    if csv_buffer:
        with open(csv_path, "w") as FF:
            FF.writelines(csv_buffer)
    if tar_track:
        Trac_out.write(''.join(json_buffer))
        Trac_out.close()
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--img-save', action='store_true', help='save the images and annotations into png directory')
    parser.add_argument('--img-save-frame', default=False, help='save the images and annotations into png directory')
    parser.add_argument('--bh-count', action='store_true', help='Count and save the behaviors')
    parser.add_argument('--chain-det', action='store_true', help='Chain_behhaviours detect')
    parser.add_argument('--tar-track', action='store_true', help='Tracking the targets')
    parser.add_argument('--tracks-show', action='store_true', help='who trackings')
    parser.add_argument('--tar-tr-start', default=1, type=int, help='default=1; The frame start to tracking')
    parser.add_argument('--path-ink', action='store_true', help='a csv file for the plot of path ink')
    parser.add_argument('--head-bind', action='store_true', help='bind the head we detected')
    parser.add_argument('--num-fly', default= 0,  type=int, help='The number of expected flies in the video')
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    parser.add_argument('--quiet', action='store_true', help='suppress per-frame progress')
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Quick screening: process only one frame every N frames (default 1 = process all frames).',
    )
    parser.add_argument(
        '--snapshot-frame',
        nargs='?',
        type=int,
        const=1,
        default=None,
        metavar='N',
        help='Single-frame mode (single video --source only): run detection on frame N only (1-based index; '
             'omit N to use the first frame). Saves annotated PNG as <video>_frame<N>_snapshot.png and YOLO-format '
             'labels/<video>_frame<N>_snapshot.txt under the run folder (class + norm xywh; add conf with --save-conf). Exits.',
    )
    parser.add_argument(
        '--frame-start',
        type=int,
        default=None,
        help='Frame window start (1-based, inclusive). Default None = first frame.',
    )
    parser.add_argument(
        '--frame-end',
        type=int,
        default=None,
        help='Frame window end (1-based, inclusive). Default None = last frame.',
    )
    parser.add_argument(
        '--tracking-dir',
        type=str,
        default='csv',
        help='Directory for tracking CSV/JSON outputs (default: csv).',
    )
    parser.add_argument(
        '--init-label-path',
        type=str,
        default=None,
        help='Optional local YOLO label file used to initialize the first tracked frame.',
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # argparse stores MISSING when flag absent; nargs='?' with const uses None only when not given
    opt.snap_frame = opt.snapshot_frame
    del opt.snapshot_frame
    print_args(FILE.stem, opt)
    return opt

#tracks-show
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
