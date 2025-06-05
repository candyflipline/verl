class AtroposNoDataFetchException(Exception):
    """
    Exception raised when no data is fetched from Atropos API.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class AtroposAPIException(Exception):
    """
    Exception raised when an error occurs while interacting with the Atropos API.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
