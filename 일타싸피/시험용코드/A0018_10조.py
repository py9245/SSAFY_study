import socket
import time
import math

# 닉네임을 사용자에 맞게 변경해 주세요.
NICKNAME = '서울18반_박유신'

# 일타싸피 프로그램을 로컬에서 실행할 경우 변경하지 않습니다.
HOST = '127.0.0.1'

# 일타싸피 프로그램과 통신할 때 사용하는 코드값으로 변경하지 않습니다.
PORT = 1447
CODE_SEND = 9901
CODE_REQUEST = 9902
SIGNAL_ORDER = 9908
SIGNAL_CLOSE = 9909


# 게임 환경에 대한 상수입니다.
TABLE_WIDTH = 254
TABLE_HEIGHT = 127
NUMBER_OF_BALLS = 6
HOLES = [[0, 0], [127, 0], [254, 0], [0, 127], [127, 127], [254, 127]]

order = 0
balls = [[0, 0] for i in range(NUMBER_OF_BALLS)]

sock = socket.socket()
print('Trying to Connect: %s:%d' % (HOST, PORT))
sock.connect((HOST, PORT))
print('Connected: %s:%d' % (HOST, PORT))

send_data = '%d/%s' % (CODE_SEND, NICKNAME)
sock.send(send_data.encode('utf-8'))
print('Ready to play!\n--------------------')


while True:

    # Receive Data
    recv_data = (sock.recv(1024)).decode()
    print('Data Received: %s' % recv_data)

    # Read Game Data
    split_data = recv_data.split('/')
    idx = 0
    try:
        for i in range(NUMBER_OF_BALLS):
            for j in range(2):
                balls[i][j] = float(split_data[idx])
                idx += 1
    except:
        send_data = '%d/%s' % (CODE_REQUEST, NICKNAME)
        print("Received Data has been currupted, Resend Requested.")
        continue

    # Check Signal for Player Order or Close Connection
    if balls[0][0] == SIGNAL_ORDER:
        order = int(balls[0][1])
        print('\n* You will be the %s player. *\n' % ('first' if order == 1 else 'second'))
        continue
    elif balls[0][0] == SIGNAL_CLOSE:
        break

    # Show Balls' Position
    print('====== Arrays ======')
    for i in range(NUMBER_OF_BALLS):
        print('Ball %d: %f, %f' % (i, balls[i][0], balls[i][1]))
    print('====================')

    angle = 0.0
    power = 0.0

    ##############################
    # 이 위는 일타싸피와 통신하여 데이터를 주고 받기 위해 작성된 부분이므로 수정하면 안됩니다.
    #
    # 모든 수신값은 변수, 배열에서 확인할 수 있습니다.
    #   - order: 1인 경우 선공, 2인 경우 후공을 의미
    #   - balls[][]: 일타싸피 정보를 수신해서 각 공의 좌표를 배열로 저장
    #     예) balls[0][0]: 흰 공의 X좌표
    #         balls[0][1]: 흰 공의 Y좌표
    #         balls[1][0]: 1번 공의 X좌표
    #         balls[4][0]: 4번 공의 X좌표
    #         balls[5][0]: 마지막 번호(8번) 공의 X좌표

    # 여기서부터 코드를 작성하세요.
    # 아래에 있는 것은 샘플로 작성된 코드이므로 자유롭게 변경할 수 있습니다.

    # whiteBall_x, whiteBall_y: 흰 공의 X, Y좌표를 나타내기 위해 사용한 변수
    whiteBall_x = balls[0][0]
    whiteBall_y = balls[0][1]

    for i in range(1, NUMBER_OF_BALLS):
        if (i == 5 or order % 2 == i % 2) and balls[i][0] >= 0:
            break

    # targetBall_x, targetBall_y: 목적구의 X, Y좌표를 나타내기 위해 사용한 변수
    targetBall_x = balls[i][0]
    targetBall_y = balls[i][1]
    ball_list = []

    for hx, hy in HOLES:
        now_angle = math.atan2(hy - targetBall_y, hx - targetBall_x)

        gx = targetBall_x - 5.73 * math.cos(now_angle)
        gy = targetBall_y - 5.73 * math.sin(now_angle)

        wh = (whiteBall_x - hx) ** 2 + (whiteBall_y - hy) ** 2
        wg = (whiteBall_x - gx) ** 2 + (whiteBall_y - gy) ** 2
        gh = (gx - hx) ** 2 + (gy - hy) ** 2

        # 조건 만족하면 후보로 등록
        if wh * 0.90 > wg + gh:
            # 각도
            radian = math.atan2(gx - whiteBall_x, gy - whiteBall_y)
            angle = (math.degrees(radian))
            if angle < 0:
                angle += 360

            # 거리
            distance = math.sqrt((gx - whiteBall_x) ** 2 + (gy - whiteBall_y) ** 2)
            power = min(distance * 0.7, 100)

            # 난이도 점수 (낮을수록 유리)
            score = wg + gh + distance

            ball_list.append((score, angle, power, gx, gy, hx, hy))

    if ball_list:
        # 난이도가 가장 낮은 샷 선택
        ball_list.sort(key=lambda x: x[0])
        best = ball_list[0]
        _, angle, power, gx, gy, hx, hy = best

        merged_data = '%f/%f/' % (angle, power)
        sock.send(merged_data.encode('utf-8'))
        print('Data Sent: %s' % merged_data)

    else:
        # 실패했을 경우 기본 샷
        merged_data = '%f/%f/' % (110, 50)
        sock.send(merged_data.encode('utf-8'))
        print('Data Sent: %s' % merged_data)
sock.close()
print('Connection Closed.\n--------------------')