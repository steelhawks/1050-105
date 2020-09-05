import logging
from controls import main_controller
import websocket
import json
import _thread as thread

logger = logging.getLogger(__name__)

def start(websocket_url):

    #websocket.enableTrace(True)
    def update_controls(ws, message):
        controls = json.loads(message)

        logger.info(controls)

        if 'request_type' in controls:

            if controls['request_type'] == 'calibration':
                main_controller.calibration = controls


        if 'controls' in controls:

            if 'camera_mode' in controls['controls']:
                main_controller.camera_mode = controls['controls']['camera_mode']


        if 'enable_calibration_feed' in controls:
            main_controller.enable_calibration_feed = controls['enable_calibration_feed']

        if 'enable_camera_feed' in controls:
            main_controller.enable_camera_feed = controls['enable_camera_feed']

        if 'enable_processing_feed' in controls:
            main_controller.enable_processing_feed = controls['enable_processing_feed']

        if 'color_profiles' in controls:
            for (camera_mode, profile ) in controls['color_profiles'].items():
                logger.info('updating %s ' % camera_mode)
                current_profile = main_controller.color_profiles.get(camera_mode)
                current_profile.update(profile)

        if 'color_profile' in controls:
            profile = controls['color_profile']
            logger.info('updating %s ' % profile['camera_mode'])
            current_profile = main_controller.color_profiles.get(profile['camera_mode'])
            current_profile.update(profile)

        #logger.info(main_controller.color_profiles)

    def ws_closed(ws):
        logger.info('closed socket')

    def on_error(ws, error):
        print(error)

    def on_open(ws):
        main_controller.enable_camera = True


    def start_dashboard_socket(*args):

        dashboard_ws = websocket.WebSocketApp(websocket_url,
            on_message = update_controls,
            on_close=ws_closed,
            on_error = on_error)
        dashboard_ws.on_open = on_open
        dashboard_ws.run_forever()

    thread.start_new_thread(start_dashboard_socket, ())
