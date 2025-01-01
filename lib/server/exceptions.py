class ServerStopException(Exception):
    def __init__(self, message: str):
        self.message = message

class NoActiveModelsException(Exception):
    def __init__(self, message: str):
        self.message = message