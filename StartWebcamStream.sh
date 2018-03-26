#!/bin/bash

# Use the following command to edit camera settings via GUI
# v4l2ucp -d /dev/video0
# v4l2ucp -d /dev/video1

# Set camera settings for both cameras
v4l2-ctl -d /dev/video0 --set-ctrl=brightness=128
v4l2-ctl -d /dev/video0 --set-ctrl=contrast=128
v4l2-ctl -d /dev/video0 --set-ctrl=saturation=255
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature_auto=0
v4l2-ctl -d /dev/video0 --set-ctrl=gain=255
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature=2000
v4l2-ctl -d /dev/video0 --set-ctrl=sharpness=255
v4l2-ctl -d /dev/video0 --set-ctrl=backlight_compensation=0
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=1000
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto_priority=0
v4l2-ctl -d /dev/video0 --set-ctrl=pan_absolute=0
v4l2-ctl -d /dev/video0 --set-ctrl=tilt_absolute=0
v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=0
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=0
v4l2-ctl -d /dev/video0 --set-ctrl=zoom_absolute=100

v4l2-ctl -d /dev/video1 --set-ctrl=brightness=128
v4l2-ctl -d /dev/video1 --set-ctrl=contrast=128
v4l2-ctl -d /dev/video1 --set-ctrl=saturation=255
v4l2-ctl -d /dev/video1 --set-ctrl=white_balance_temperature_auto=0
v4l2-ctl -d /dev/video1 --set-ctrl=gain=255
v4l2-ctl -d /dev/video1 --set-ctrl=white_balance_temperature=2000
v4l2-ctl -d /dev/video1 --set-ctrl=sharpness=255
v4l2-ctl -d /dev/video1 --set-ctrl=backlight_compensation=0
v4l2-ctl -d /dev/video1 --set-ctrl=exposure_auto=1
v4l2-ctl -d /dev/video1 --set-ctrl=exposure_absolute=1000
v4l2-ctl -d /dev/video1 --set-ctrl=exposure_auto_priority=0
v4l2-ctl -d /dev/video1 --set-ctrl=pan_absolute=0
v4l2-ctl -d /dev/video1 --set-ctrl=tilt_absolute=0
v4l2-ctl -d /dev/video1 --set-ctrl=focus_auto=0
v4l2-ctl -d /dev/video1 --set-ctrl=focus_absolute=0
v4l2-ctl -d /dev/video1 --set-ctrl=zoom_absolute=100

# If entered a, display webcam from /dev/video0
if [ $1 = "a" ]
then
    cvlc --no-audio v4l2:///dev/video0:width=640:height=360 --v4l2-chroma MJPG  --sout '#standard{access=http,mux=mpjpeg,host=128.40.57.144,dst=:8554/}'
fi

# If entered b, display webcam from /dev/video1
if [ $1 = "b" ]
then
    cvlc --no-audio v4l2:///dev/video1:width=640:height=360 --v4l2-chroma MJPG  --sout '#standard{access=http,mux=mpjpeg,host=128.40.57.144,dst=:8555/}'
fi

# If entered ab, display both cameras
if [ $1 = "ab" ]
then
    cvlc --vlm-conf ~/openEPhys_DACQ/StartWebcamStream.vlm.conf --mosaic-width 640 --mosaic-height 720 --mosaic-order "1,2"
fi

