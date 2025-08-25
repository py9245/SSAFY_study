import os

import google.generativeai as genai


def run_gemini():
   try:
        # API 키 설정
   # 터미널에서 설정한 환경 변수로부터 API 키를 불러옵니다.

      api_key = os.getenv("GEMINI_API_KEY")

      if not api_key:
         print("오류: GEMINI_API_KEY 환경 변수를 찾을 수 없습니다.")
         print("API 키를 설정했는지 다시 확인해주세요.")
         return

      genai.configure(api_key=api_key)

     # 모델 초기화
     # 'gemini-pro'는 텍스트 기반 질문에 적합한 모델입니다.

      model = genai.GenerativeModel('gemini-pro')

      prompt = "파이썬으로 만들 수 있는 간단한 프로그램 3가지 추천해줘"

      print(f"요청: \"{prompt}\"")

      print("-" * 20)

      response = model.generate_content(prompt)

      print("[Gemini 응답]")
      print(response.text)

   except Exception as e:
      print(f"API 사용 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    run_gemini()
