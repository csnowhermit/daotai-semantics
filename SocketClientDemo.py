import time
import socket

'''
    socket client：模拟语音端过来的输入
'''

p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
p.connect(('192.168.0.27', 50007))
while True:
    msg = input('please input:')
    # 防止输入空消息
    if not msg:
        continue

    msgDict = {}
    msgDict["daotaiID"] = "center01"
    msgDict["sentences"] = msg
    msgDict["timestamp"] = int(time.time())
    msgDict["msgCalled"] = "onResult"

    p.send(str(msgDict).encode('utf-8'))  # 收发消息一定要二进制，记得编码
    if msg == '1':
        break
p.close()