import socket
server = socket.socket()
server.bind(("localhost",6969))
server.listen(5)

# 接入多个客户端
while True:
    conn,addr = server.accept()
    print(conn,addr)

    count = 0
    while True:
        data = conn.recv(1024)
        print("data:", data)
        # 单个客户端退出
        if not data:
            print("client is lost...")
            break
        conn.send(data)
        count += 1
        if count > 10:
            break
server.close()