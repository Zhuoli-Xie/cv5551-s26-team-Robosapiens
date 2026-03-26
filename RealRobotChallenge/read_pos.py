from xarm.wrapper import XArmAPI

arm = XArmAPI('192.168.1.168')  # 你的机械臂IP
arm.connect()

code, pos = arm.get_position()

print('code:', code)
print('position:', pos)
