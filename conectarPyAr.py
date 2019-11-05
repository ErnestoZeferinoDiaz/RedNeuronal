import serial, time

arduino = serial.Serial('COM3', 9600)
for i in range(10):
    time.sleep(2)
    rawString = arduino.readline()
    print(rawString)
arduino.close()