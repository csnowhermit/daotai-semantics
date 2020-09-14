
sentences = '''
    {"daotaiID":"center01","message":"driver.IatActivity2Semantics session id =null","msgCalled":"onEvent","timestamp":1600049538073}{"daotaiID":"center01","message":"你。","msgCalled":"onResult","timestamp":1600049538078}{"daotaiID":"center01","message":"driver.IatActivity2Semantics onBeginOfSpeech","msgCalled":"onBeginOfSpeech","timestamp":1600049538082}
    '''

# recvJson = eval(sentences)    # 这样只能处理单个json发来的清空
# print(recvJson)

before_braces = []
end_braces = []

for index, s in enumerate(sentences):
    if sentences[index] == "{":
        before_braces.append(index)
    elif sentences[index] == "}":
        end_braces.append(index)

print(before_braces)
print(end_braces)

for i, j in zip(before_braces, end_braces):
    print(i, j)
    # print()
    recvStr = sentences[i: j+1]
    recvJson = eval(recvStr)
    print(recvJson)