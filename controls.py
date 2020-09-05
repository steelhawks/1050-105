CAMERA_MODE_RAW = 'RAW'
CAMERA_MODE_CALIBRATE = 'CALIBRATE'
CAMERA_MODE_BALL = 'BALL'
CAMERA_MODE_HEXAGON = 'HEXAGON'
CAMERA_MODE_LOADING_BAY = 'BAY'

class Controls():

    def __init__(self):
        self.enable_camera = True

        self.enable_camera_feed = True
        self.enable_calibration_feed = False
        self.enable_processing_feed = True
        self.enable_dual_camera = False
        self.send_tracking_data = True

        self.camera_mode = CAMERA_MODE_LOADING_BAY
        self.enable_feed = True
        self.color_profiles = {}

        self.calibration = {}

    def update(message):
        print(message)

main_controller = Controls()
