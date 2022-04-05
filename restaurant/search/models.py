from django.db import models

# Create your models here.

class Place:
    HOS = 3
    COL = 3.5
    STD = 1
    THR = 1.5
    AUD = 1
    FAM = 2.5
    DEN = 2
    name: str
    rank: int
    health_centre: int
    college: int
    stadium: int
    theatre: int
    auditorium: int
    home: int
    bus: str
    railway: str
    pop_den: float
