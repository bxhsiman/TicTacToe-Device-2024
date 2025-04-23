import serial

ser = serial.Serial('COM6', 9600)
ser.write('c1'+'b5')
ser.close()
