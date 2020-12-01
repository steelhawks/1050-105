import json
import logging
from os.path import abspath, dirname, join, exists

root_path = abspath(join(dirname(__file__)))

logger = logging.getLogger(__name__)

class ColorProfileEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, ColorProfile):
      return obj.to_encodable()
    return json.JSONEncoder.default(self, obj)

class Range:

  def __init__(self, min, max):
    self.min = min
    self.max = max


class ColorProfile:

  def __init__(self, camera_mode):

    self.camera_mode = camera_mode

    self.red = Range(0,255)
    self.green = Range(0,255)
    self.blue = Range(0,255)

    self.hsl_hue = Range(0,255)
    self.hsl_sat = Range(0,255)
    self.hsl_lum = Range(0,255)

    self.hsv_hue = Range(0,255)
    self.hsv_sat = Range(0,255)
    self.hsv_val = Range(0,255)


    camera_mode_filepath = join(root_path, 'color_profile_%s.json' % camera_mode)

    if exists(camera_mode_filepath):
      logger.info("loading from %s" % camera_mode_filepath)
      with open(camera_mode_filepath, mode="rb" ) as f:
        profile = json.loads(f.read())
        self.update(profile)

  def update(self, profile):

    color_profile = self

    color_profile.red.min = int(profile['rgb']['r']['min'])
    color_profile.red.max = int(profile['rgb']['r']['max'])

    color_profile.green.min = int(profile['rgb']['g']['min'])
    color_profile.green.max = int(profile['rgb']['g']['max'])

    color_profile.blue.min = int(profile['rgb']['b']['min'])
    color_profile.blue.max = int(profile['rgb']['b']['max'])

    color_profile.hsv_hue.min = int(profile['hsv']['h']['min'])
    color_profile.hsv_hue.max = int(profile['hsv']['h']['max'])

    color_profile.hsv_sat.min = int(profile['hsv']['s']['min'])
    color_profile.hsv_sat.max = int(profile['hsv']['s']['max'])

    color_profile.hsv_val.min = int(profile['hsv']['v']['min'])
    color_profile.hsv_val.max = int(profile['hsv']['v']['max'])

    color_profile.hsl_hue.min = int(profile['hsl']['h']['min'])
    color_profile.hsl_hue.max = int(profile['hsl']['h']['max'])

    color_profile.hsl_sat.min = int(profile['hsl']['s']['min'])
    color_profile.hsl_sat.max = int(profile['hsl']['s']['max'])

    color_profile.hsl_lum.min = int(profile['hsl']['l']['min'])
    color_profile.hsl_lum.max = int(profile['hsl']['l']['max'])


  def to_encodable(self):
    rgb = dict(
      r=dict(min=self.red.min,
                     max=self.red.max),
      b=dict(min=self.blue.min,
                     max=self.blue.max),
      g=dict(min=self.green.min,
                     max=self.green.max)
    )

    hsl = dict(
      h=dict(min=self.hsl_hue.min,
                     max=self.hsl_hue.max),
      s=dict(min=self.hsl_sat.min,
                     max=self.hsl_sat.max),
      l=dict(min=self.hsl_lum.min,
                     max=self.hsl_lum.max)
    )

    hsv = dict(
      h=dict(min=self.hsv_hue.min,
          max=self.hsv_hue.max),
      s=dict(min=self.hsv_sat.min,
          max=self.hsv_sat.max),
      v=dict(min=self.hsv_val.min,
          max=self.hsv_val.max)
    )

    return dict(camera_mode=self.camera_mode,
          rgb=rgb,
          hsv=hsv,
          hsl=hsl)
