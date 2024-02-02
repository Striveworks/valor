class ClientException(Exception):
    def __init__(self, resp):
        self.status_code = resp.status_code
        self.detail = resp.json()["detail"]
        super().__init__(str(self.detail))


class ClientAlreadyConnectedError(Exception):
    def __init__(self):
        super().__init__("Client already connected.")


class ClientNotConnectedError(Exception):
    def __init__(self):
        super().__init__("Client not connected.")


class ClientConnectionFailed(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)
