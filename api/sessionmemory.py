from cachetools import TTLCache
from utils import Uid

class SessionMemory:
    @staticmethod
    def init():
        # Cache de 100 éléments max, TTL de 300 secondes
        SessionMemory.cache = TTLCache(maxsize=100, ttl=3600)

    @staticmethod
    def getSession(sessionId:str):
        return SessionMemory.cache.get(str(sessionId))
    
    @staticmethod
    def createSession(data:dict):
        uid = Uid.create()
        SessionMemory.cache[str(uid)] = data
        return uid
    
    @staticmethod
    def updateSession(sessionId:str, data:dict):
        SessionMemory.cache[str(sessionId)] = data