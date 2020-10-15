import socket
client = socket.socket()
# 连接
client.connect(("localhost",6969))

# 发送信息
while True:
    data = input(">>:").strip()
    print(len(data))
    if len(data) == 0:
        continue
    send_data = data.encode("utf-8")
    client.send(send_data)
    # 接收信息
    data = client.recv(10240)
    print("recv:",data.decode("utf-8"))
client.close()