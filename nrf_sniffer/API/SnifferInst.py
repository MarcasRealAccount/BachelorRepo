from API.Sniffer import FindSniffers, Sniffer

from threading import Thread
from collections.abc import Callable

class SnifferInst:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SnifferInst, cls).__new__(cls)
        return cls.instance
    
    def __init__(self):
        self._sniffer:Sniffer|None = getattr(self, "_sniffer", None)
        self._connectThread:Thread = getattr(self, "_connectThread", None)

    def __del__(self):
        self.Close()

    def TryConnect(self, notify:Callable|None = None):
        """
        Try to connect to a sniffer.
        Spawns up a background thread to make the attempt.
        """
        
        if self.IsAlive() or self._connectThread is not None:
            return
        
        self._connectThread = Thread(target=self._connect, name="Connector", args=(notify,), daemon=True)
        self._connectThread.start()
    
    def IsAlive(self) -> bool:
        """
        Is the sniffer alive? I.e. do we have a connection with a sniffer?
        """
        return self._sniffer is not None and self._sniffer.isAlive()

    def Close(self):
        """
        Close the sniffer.
        """
        if self.IsAlive():
            self._sniffer.close()

    def _connect(self, notify:Callable|None):
        sniffers = FindSniffers()
        for snifferInfo in sniffers:
            sniffer = Sniffer(snifferInfo)
            if sniffer.isAlive():
                self._sniffer = sniffer
                break
        if notify is not None and self.IsAlive():
            notify()
        self._connectThread = None

    @property
    def sniffer(self):
        return self._sniffer