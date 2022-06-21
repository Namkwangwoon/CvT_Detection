from .default import _C as config
from .default import update_config
from .default import _update_config_from_file
from .default import save_config

import config.global_settings as settings

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))

settings = Settings(settings)