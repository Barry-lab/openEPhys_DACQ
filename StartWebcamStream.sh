#!/bin/bash
v4l2-ctl --set-ctrl=brightness=128
v4l2-ctl --set-ctrl=contrast=128
v4l2-ctl --set-ctrl=saturation=255
v4l2-ctl --set-ctrl=white_balance_temperature_auto=0
v4l2-ctl --set-ctrl=gain=255
v4l2-ctl --set-ctrl=white_balance_temperature=2000
v4l2-ctl --set-ctrl=sharpness=255
v4l2-ctl --set-ctrl=backlight_compensation=0
v4l2-ctl --set-ctrl=exposure_auto=1
v4l2-ctl --set-ctrl=exposure_absolute=1000
v4l2-ctl --set-ctrl=exposure_auto_priority=0
v4l2-ctl --set-ctrl=pan_absolute=0
v4l2-ctl --set-ctrl=tilt_absolute=0
v4l2-ctl --set-ctrl=focus_auto=0
v4l2-ctl --set-ctrl=focus_absolute=0
v4l2-ctl --set-ctrl=zoom_absolute=100

cvlc --no-audio v4l2:///dev/video0:width=1280:height=720 --v4l2-chroma MJPG  --sout '#standard{access=http,mux=mpjpeg,host=128.40.57.144,dst=:8554/}'

#v4l2ucp
