# vehicle_violation_detection

For most of the following files, you just need to click the run button to execute them. 
If any command needs to be entered manually, I will indicate it.

- ### UI Interface:
  Run this file and it will complete all the operations automatically.
  + ./UI_interface/UI.py

- ### Vehicle Detection:
    + ./YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect/predict.py
  [predict.py](YOLOv8-DeepSORT-Object-Tracking%2Fultralytics%2Fyolo%2Fv8%2Fdetect%2Fpredict.py)
  ```
  cd ./YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect/
  ```
  ```
  python predict.py model=yolov8l.pt source="your_file_name.mp4 (or .mov)" show=True
  ```
    + ./vehicle/vehicle_detection_video_deepsort.py
  [vehicle_detection_video_deepsort.py](vehicle%2Fvehicle_detection_video_deepsort.py)
- ### Blinker  Detection:
    + ./vehicle/blinker_detection.py
  [blinker_detection.py](vehicle%2Fblinker_detection.py)
- ### Violation Determination:
    + ./violation_determination/my_scripts/houghlines_merge.py
  [houghlines_merge.py](violation_determination%2Fmy_scripts%2Fhoughlines_merge.py)
    + ./violation_determination/my_scripts/merge_similar_lane_detections.py
  [merge_similar_lane_detections.py](violation_determination%2Fmy_scripts%2Fmerge_similar_lane_detections.py)
- ### Vehicle in Which Lane:
    + ./violation_determination/my_scripts/car_in_which_lane.py
  [car_in_which_lane.py](violation_determination%2Fmy_scripts%2Fcar_in_which_lane.py)
- ### Violation Detection:
    + ./violation_determination/my_scripts/determine_blinkers_violation.py\
[determine_blinkers_violation.py](violation_determination%2Fmy_scripts%2Fdetermine_blinkers_violation.py)
    + ./violation_determination/tools/demo.py [demo.py](violation_determination%2Ftools%2Fdemo.py)
- ### Data:
    + ./violation_determination/inference
  [inference](violation_determination%2Finference)
    + ./vehicle/data
  [data](vehicle%2Fdata)
    + ./YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect/videos
  [videos](YOLOv8-DeepSORT-Object-Tracking%2Fultralytics%2Fyolo%2Fv8%2Fdetect%2Fvideos)