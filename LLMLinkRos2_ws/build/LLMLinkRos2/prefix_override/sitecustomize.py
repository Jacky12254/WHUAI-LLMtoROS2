import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jacky/LLMLinkRos2_ws/install/LLMLinkRos2'
