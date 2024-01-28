# Lane Detection 
that finds distance from center of detected lanes.
- warp2.py and live_warp.py work with car lanes.
- **ONLY** agri_warp.py and live_agri_warp.py work for agricultural rows of crops

https://github.com/PatD123/lane_detec/assets/76749942/dda1d859-2e88-408e-854d-e4534c311b26

# How to use?
1. Make python virtual env and activate venv.
2. Install requirements.
3. Change directory into /lane_detec/src
4. Run the desired .py file.

For individual frames...
```
py agri_warp.py
```
For full videos...
```
py live_agri_warp.py
```

# Goals!!!!
1. Kalman Filtering to track lines more smoothly
2. Find a way to determine when to switch ROIs

> [!NOTE] 
If anybody can tell me about 2, that'd be cool. But no one looks
at this repo, unfortunado.... 
