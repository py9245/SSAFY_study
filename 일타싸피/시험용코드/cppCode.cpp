#define _USE_MATH_DEFINES
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <winsock2.h>
#include <windows.h>

#pragma comment(lib, "ws2_32.lib")

// 닉네임을 사용자에 맞게 변경해 주세요.
#define NICKNAME "C++플레이어"

// 일타싸피 프로그램을 로컬에서 실행할 경우 변경하지 않습니다.
#define HOST "127.0.0.1"
#define PORT 1447

// 일타싸피 프로그램과 통신할 때 사용하는 코드값으로 변경하지 않습니다.
#define CODE_SEND 9901
#define CODE_REQUEST 9902
#define SIGNAL_ORDER 9908
#define SIGNAL_CLOSE 9909

void ErrorHandling(const std::string& message);
std::vector<std::vector<float>> ParseReceivedData(const std::string& data);
std::string ConvertToUTF8(const std::wstring& wstr);

int main()
{
	// 게임 환경에 대한 상수입니다.
	const int TABLE_WIDTH = 254;
	const int TABLE_HEIGHT = 127;
	const int NUMBER_OF_BALLS = 6;
	const int HOLES[6][2] = { {0, 0}, {127, 0}, {254, 0}, {0, 127}, {127, 127}, {254, 127} };

	std::vector<std::vector<float>> balls(NUMBER_OF_BALLS, std::vector<float>(2, 0.0f));
	int order = 0;

	WSADATA wsaData;
	SOCKET hSocket;
	SOCKADDR_IN sockAddr;

	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
		ErrorHandling("WSAStartup failure.");

	hSocket = socket(PF_INET, SOCK_STREAM, 0);
	if (hSocket == INVALID_SOCKET)
		ErrorHandling("Socket Creating Failure.");

	memset(&sockAddr, 0, sizeof(sockAddr));
	sockAddr.sin_family = AF_INET;
	sockAddr.sin_addr.s_addr = inet_addr(HOST);
	sockAddr.sin_port = htons(PORT);

	std::cout << "Trying Connect: " << HOST << ":" << PORT << std::endl;
	if (connect(hSocket, (SOCKADDR*)&sockAddr, sizeof(sockAddr)) == SOCKET_ERROR)
		ErrorHandling("Connection Failure.");
	else
		std::cout << "Connected: " << HOST << ":" << PORT << std::endl;

	std::wstring wnickname = L"" NICKNAME "";
	std::string utf8Nickname = ConvertToUTF8(wnickname);
	std::string sendData = std::to_string(CODE_SEND) + "/" + utf8Nickname + "/";
	send(hSocket, sendData.c_str(), sendData.size(), 0);
	std::cout << "Ready to play!\n--------------------\n";

	while (true)
	{
		// Receive Data
		char message[1024];
		int strLen = recv(hSocket, message, sizeof(message) - 1, 0);
		if (strLen <= 0) break;

		message[strLen] = '\0';
		std::string receivedData(message);

		// Read Game Data
		balls = ParseReceivedData(receivedData);

		// Check Signal for Player Order or Close Connection
		if (balls[0][0] == SIGNAL_ORDER)
		{
			order = static_cast<int>(balls[0][1]);
			std::cout << "\n\n* You will be the " << (order == 1 ? "first" : "second") << " player. *\n\n";
			continue;
		}
		else if (balls[0][0] == SIGNAL_CLOSE)
		{
			break;
		}

		float angle = 0.0f;
		float power = 0.0f;

		//////////////////////////////
		// 이 위는 일타싸피와 통신하여 데이터를 주고 받기 위해 작성된 부분이므로 수정하면 안됩니다.
		//
		// 모든 수신값은 변수, 배열에서 확인할 수 있습니다.
		//   - order: 1인 경우 선공, 2인 경우 후공을 의미
		//   - balls[][]: 일타싸피 정보를 수신해서 각 공의 좌표를 배열로 저장
		//     예) balls[0][0]: 흰 공의 X좌표
		//         balls[0][1]: 흰 공의 Y좌표
		//         balls[1][0]: 1번 공의 X좌표
		//         balls[4][0]: 4번 공의 X좌표
		//         balls[5][0]: 마지막 번호(8번) 공의 X좌표

		// 여기서부터 코드를 작성하세요.
		// 아래에 있는 것은 샘플로 작성된 코드이므로 자유롭게 변경할 수 있습니다.




		// whiteBall_x, whiteBall_y: 흰 공의 X, Y좌표를 나타내기 위해 사용한 변수
		float whiteBall_x = balls[0][0];
		float whiteBall_y = balls[0][1];

		// targetBall_x, targetBall_y: 목적구의 X, Y좌표를 나타내기 위해 사용한 변수
		float targetBall_x = balls[1][0];
		float targetBall_y = balls[1][1];

		// width, height: 목적구와 흰 공의 X좌표 간의 거리, Y좌표 간의 거리
		float width = std::fabs(targetBall_x - whiteBall_x);
		float height = std::fabs(targetBall_y - whiteBall_y);

		// 흰 공과 목적구의 상대 위치에 따른 각도 설정
		double angleTemp = (whiteBall_x == targetBall_x) ? (whiteBall_y < targetBall_y ? 0 : 180)
			: (whiteBall_y == targetBall_y ? (whiteBall_x < targetBall_x ? 90 : 270)
				: (whiteBall_x > targetBall_x && whiteBall_y > targetBall_y ? atan(width / height) * 180 / M_PI + 180
					: (whiteBall_x < targetBall_x && whiteBall_y > targetBall_y ? atan(height / width) * 180 / M_PI + 90
						: atan(width / height) * 180 / M_PI)));

		angle = (float)angleTemp;
		
		// distance: 두 점(좌표) 사이의 거리를 계산
		double distance = std::sqrt((width * width) + (height * height));
		
		// power: 거리 distance에 따른 힘의 세기를 계산
		power = static_cast<float>(distance) * 0.5f;




		// 주어진 데이터(공의 좌표)를 활용하여 두 개의 값을 최종 결정하고 나면,
		// 나머지 코드에서 일타싸피로 값을 보내 자동으로 플레이를 진행하게 합니다.
		//   - angle: 흰 공을 때려서 보낼 방향(각도)
		//   - power: 흰 공을 때릴 힘의 세기
		// 
		// 이 때 주의할 점은 power는 100을 초과할 수 없으며,
		// power = 0인 경우 힘이 제로(0)이므로 아무런 반응이 나타나지 않습니다.
		//
		// 아래는 일타싸피와 통신하는 나머지 부분이므로 수정하면 안됩니다.
		//////////////////////////////

		std::string mergedData = std::to_string(angle) + "/" + std::to_string(power) + "/";
		send(hSocket, mergedData.c_str(), mergedData.size(), 0);
		std::cout << "Data Sent: " << mergedData << std::endl;
	}

	closesocket(hSocket);
	WSACleanup();
	return 0;
}

void ErrorHandling(const std::string& message)
{
	std::cerr << message << std::endl;
	exit(1);
}

std::vector<std::vector<float>> ParseReceivedData(const std::string& data)
{
	std::vector<std::vector<float>> balls(6, std::vector<float>(2, 0.0f));
	size_t pos = 0;
	size_t index = 0;
	std::string token;
	std::string tempData = data;

	while ((pos = tempData.find('/')) != std::string::npos && index < 12)
	{
		token = tempData.substr(0, pos);
		balls[index / 2][index % 2] = std::stof(token);
		tempData.erase(0, pos + 1);
		index++;
	}
	return balls;
}

std::string ConvertToUTF8(const std::wstring& wstr)
{
	int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
	std::string utf8Str(size_needed, 0);
	WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &utf8Str[0], size_needed, NULL, NULL);
	return utf8Str;
}
