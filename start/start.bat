echo "1."
cmd /k "cd /d D:\workspace\workspace_python/daotai-semantics&&activate dlp&&pythonw rabbitmqConsumer.py"
echo "2."
cmd /k "cd /d D:\workspace\workspace_python/daotai-semantics&&activate dlp&&pythonw portrait_reciver.py"
echo "3."
cmd /k "cd /d D:\workspace\workspace_python/daotai-semantics/bayes&&activate dlp&&pythonw bayes_test_from_socket.py"
echo "4."
cmd /k "cd /d D:\workspace\workspace_python/daotai-semantics&&activate dlp&&pythonw percept_coming_cstack.py"
echo "5."
cmd /k "cd /d D:\workspace\workspace_python/daotai-semantics&&activate dlp&&pythonw iat_command_receiver.py"