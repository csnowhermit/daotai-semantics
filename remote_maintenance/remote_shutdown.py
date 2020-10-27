#coding=utf-8
import os
import sys
from flask import Flask
from flask import request

base_path = "D:/workspace/workspace_python/daotai-semantics/"
sys.path.append(base_path)
from utils.dbUtil import saveRemoteCmd2DB

app = Flask(__name__)

@app.route("/login", methods=["GET", "POST"])
def hello_str():
    if request.method == "POST":
        cmd = request.form.get("cmd")
    else:
        cmd = request.args.get("cmd")
    print("cmd:", cmd)
    saveRemoteCmd2DB(cmd)
    if cmd is not None or len(cmd) > 0:
        os.system(cmd)
    return cmd

if __name__ == "__main__":
    app.run(host='192.168.0.34', port=8000, debug=True)