import sys
import socket
import time
import traceback
from utils.dbUtil import getCurrDateStatus

sys.path.append("..")
from Logger import *

# 日志
iatcommand_logfile = 'D:/data/daotai_iatcommand.log'
iatcommand_log = Logger(iatcommand_logfile, level='info')

def start():
    sev = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP连接
    HOST, PORT = "192.168.0.27", 50009
    sev.bind((HOST, PORT))
    sev.listen()
    print("iat命令接收器已启动。。。")
    iatcommand_log.logger.info("iat命令接收器已启动。。。")

    conn, addr = sev.accept()
    print(conn, addr)
    iatcommand_log.logger.info("%s %s" % (conn, addr))
    empty_package_nums = 0    # 记录空包的数量

    while True:
        try:
            recvStr = bytes.decode(conn.recv(4096), encoding='utf-8')
            if len(recvStr) == 0:    # 如果是安卓客户端，当客户端断开时，服务端收到的是空包
                empty_package_nums += 1
                if empty_package_nums >= 200:
                    raise ConnectionResetError
                continue
            else:
                empty_package_nums = 0    # 如果遇到非空包来，则空包数量重新计数

            status = getCurrDateStatus()
            print("指令：%s, 引擎：%s" % (recvStr, status))
            iatcommand_log.logger.info("指令：%s, 引擎：%s" % (recvStr, status))
            if status == 0:    # 为0，说明能用在线版
                p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                p.connect(("192.168.0.27", 50008))

                # 将指令做透传
                p.send(str(recvStr).encode('utf-8'))  # 收发消息一定要二进制，记得编码
                p.close()
            else:    # 否则只能用离线版
                p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                p.connect(("192.168.0.144", 50008))

                p.send(str(recvStr).encode('utf-8'))  # 收发消息一定要二进制，记得编码
                p.close()

        except ConnectionResetError as connectionResetError:
            iatcommand_log.logger.warn("客户端已断开，正在等待重连: %s" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
            print("客户端已断开，正在等待重连: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            conn, addr = sev.accept()
            print(conn, addr)
            iatcommand_log.logger.info("%s %s" % (conn, addr))
            continue
        except Exception as e:
            traceback.print_exc(file=open(iatcommand_logfile, 'a+'))
            continue

if __name__ == '__main__':
    start()