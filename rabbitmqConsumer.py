from portrait_reciver import getRabbitConn

'''
    rabbitmq consumer demo
'''

# 定义一个回调函数来处理，这边的回调函数就是将信息打印出来。
def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)

if __name__ == '__main__':
    consumer_channel, consumer_EXCHANGE_NAME, consumer_queueName, consumer_routingKey = getRabbitConn("rabbit2backstage")

    print("rabbit 已启动：%s %s %s %s" % (consumer_channel, consumer_EXCHANGE_NAME, consumer_queueName, consumer_routingKey))

    consumer_channel.basic_consume(queue=consumer_queueName, on_message_callback=callback,
                                   auto_ack=True)  # 这里写的是QUEUE_NAME，而不是routingKey

    print(' [*] Waiting for messages. To exit press CTRL+C')

    # 开始接收信息，并进入阻塞状态，队列里有信息才会调用callback进行处理。按ctrl+c退出。
    consumer_channel.start_consuming()
