import configparser
import pika
import threading
# from portrait_reciver import getRabbitConn

'''
    rabbitmq consumer demo
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
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, virtual_host=vhost, credentials=credentials))
    connection.process_data_events()    # 防止主进程长时间等待，而导致rabbitmq主动断开连接，所以要定期发心跳调用
    channel = connection.channel()
    channel.exchange_declare(exchange=EXCHANGE_NAME,
                             exchange_type='direct')    # 声明交换机
    channel.queue_declare(queue=queueName)    # 声明队列。消费者需要这样代码，生产者不需要
    channel.queue_bind(queue=queueName, exchange=EXCHANGE_NAME, routing_key=routingKey)    # 绑定队列和交换机

    return channel, EXCHANGE_NAME, queueName, routingKey

# 定义一个回调函数来处理，这边的回调函数就是将信息打印出来。
def callback_rabbit2backstage(ch, method, properties, body):
    recvStr = bytes.decode(body, encoding='utf-8')
    print(" [x] rabbit2backstage %r" % recvStr)

# 定义一个回调函数来处理，这边的回调函数就是将信息打印出来。
def callback_rabbit2portrait(ch, method, properties, body):
    recvStr = bytes.decode(body, encoding='utf-8')
    print(" [x] rabbit2portrait %r" % recvStr)


def rabbit2backstage():
    consumer_channel, consumer_EXCHANGE_NAME, consumer_queueName, consumer_routingKey = getRabbitConn("rabbit2backstage")
    print("rabbit2backstage 已启动：%s %s %s %s" % (consumer_channel, consumer_EXCHANGE_NAME, consumer_queueName, consumer_routingKey))

    consumer_channel.basic_consume(queue=consumer_queueName, on_message_callback=callback_rabbit2backstage, auto_ack=True)  # 这里写的是QUEUE_NAME，而不是routingKey
    print(' [rabbit2backstage] Waiting for messages. To exit press CTRL+C')

    # 开始接收信息，并进入阻塞状态，队列里有信息才会调用callback进行处理。按ctrl+c退出。
    consumer_channel.start_consuming()

def rabbit2portrait():
    consumer_channel, consumer_EXCHANGE_NAME, consumer_queueName, consumer_routingKey = getRabbitConn("rabbit2portrait")
    print("rabbit2portrait 已启动：%s %s %s %s" % (consumer_channel, consumer_EXCHANGE_NAME, consumer_queueName, consumer_routingKey))

    consumer_channel.basic_consume(queue=consumer_queueName, on_message_callback=callback_rabbit2portrait,
                                   auto_ack=True)  # 这里写的是QUEUE_NAME，而不是routingKey
    print(' [rabbit2portrait] Waiting for messages. To exit press CTRL+C')

    # 开始接收信息，并进入阻塞状态，队列里有信息才会调用callback进行处理。按ctrl+c退出。
    consumer_channel.start_consuming()

if __name__ == '__main__':
    backstage = threading.Thread(target=rabbit2backstage)
    backstage.start()

    portrait = threading.Thread(target=rabbit2portrait)
    portrait.start()
