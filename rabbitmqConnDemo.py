import time
import configparser
import pika


'''
    获取rabbitmq连接
    :param nodeName 指定配置文件的哪个节点
'''
def getRabbitConn(nodeName):
    cf = configparser.ConfigParser()
    cf.read("./kdata/config.conf")
    host = str(cf.get(nodeName, "host"))
    port = int(cf.get(nodeName, "port"))
    username = str(cf.get(nodeName, "username"))
    password = str(cf.get(nodeName, "password"))
    EXCHANGE_NAME = str(cf.get(nodeName, "EXCHANGE_NAME"))
    vhost = str(cf.get(nodeName, "vhost"))
    routingKey = str(cf.get(nodeName, "routingKey"))
    queueName = str(cf.get(nodeName, "QUEUE_NAME"))

    credentials = pika.PlainCredentials(username=username, password=password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, heartbeat=0, virtual_host=vhost, credentials=credentials))
    connection.process_data_events()    # 防止主进程长时间等待，而导致rabbitmq主动断开连接，所以要定期发心跳调用
    channel = connection.channel()
    # channel.queue_declare(queue=routingKey, durable=True)    # 定义持久化队列
    # channel.queue_declare(queue=routingKey)  # 定义持久化队列

    # 以下原来是在消费者里面
    channel.exchange_declare(exchange=EXCHANGE_NAME,
                             exchange_type='direct')    # 声明交换机
    channel.queue_declare(queue=queueName)    # 声明队列。消费者需要这样代码，生产者不需要
    channel.queue_bind(queue=queueName, exchange=EXCHANGE_NAME, routing_key=routingKey)    # 绑定队列和交换机

    return connection, channel, EXCHANGE_NAME, routingKey


if __name__ == '__main__':
    list = []
    for i in range(10000):
        start = time.time()
        s = "Hello-%s" % (str(i))
        # print(s)
        backstage_connection, backstage_channel, backstage_EXCHANGE_NAME, backstage_routingKey = getRabbitConn("rabbit2backstage")
        backstage_channel.basic_publish(exchange=backstage_EXCHANGE_NAME,
                                        routing_key=backstage_routingKey,
                                        body=s)  # 将语义识别结果给到后端
        end = time.time()
        print(end-start)
        list.append(end-start)

    print("mean:", float(sum(list)/len(list)))
    print("max:", float(max(list)))
    print("min:", float(min(list)))


