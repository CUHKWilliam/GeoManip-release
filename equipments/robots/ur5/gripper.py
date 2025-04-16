import serial
from time import sleep

class Gripper():
    def __init__(self,gripper_flag=True):
        self.init_gripper(gripper_flag)
        # self.gripper_open()
        
    def init_gripper(self, gripper_flag):
        if gripper_flag:
            self.serialPort = serial.Serial(
            port="/dev/ttyUSB0", 
            baudrate=9600, 
            bytesize=8, 
            timeout=0.1, 
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_ODD,
            )     
        return None

    def gripper_open(self):
        input_b = int("1000",2)
        self.serialPort.write(bytes([input_b]))
        msg = self.serialPort.readline()
        sleep(3)

    def gripper_close(self):
        input_b = int("0000",2)
        self.serialPort.write(bytes([input_b]))
        msg = self.serialPort.readline()
        sleep(3)