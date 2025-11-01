# Codex CLI에서 Notion MCP 연동 가이드

## 개요
이 문서는 Codex CLI 환경에서 Notion MCP를 연결하면서 진행했던 작업을 정리한 것입니다. `rmcp_client` 기능 활성화, Notion MCP 서버 빌드, 토큰 설정, 그리고 초기화 오류 해결 과정이 포함되어 있습니다.

## 사전 준비
- Node.js와 npm이 설치되어 있어야 합니다. (`npm --version`, `node --version`으로 확인)
- Codex CLI 구성 파일은 `C:\Users\유신\.codex\config.toml` 기준으로 설명합니다.
- Notion에서 발급한 **Internal Integration Secret**(예: `ntn_...`)을 준비합니다.
- `.codex\notion-mcp-server` 폴더는 `npx -y @notionhq/notion-mcp-server`를 실행하면 Codex가 자동으로 내려받아 두는 위치입니다. 이미 존재한다면 그대로 사용하면 되고, 없다면 위 명령으로 생성한 뒤 이 문서를 따라 진행하면 됩니다.

## 단계별 설정

### 1. `config.toml` 기본 설정 확인
```toml
[features]
rmcp_client = true

[mcp_servers.notion]
command = "node"
args = ["C:\\Users\\유신\\.codex\\notion-mcp-server\\bin\\cli.mjs"]

[mcp_servers.notion.env]
NOTION_TOKEN = 노션 api
```
- `experimental_use_rmcp_client` 항목은 제거했습니다. `rmcp_client = true`만으로 충분합니다.
- `command`를 `node`, `args`를 로컬에 빌드된 `cli.mjs` 경로로 지정해 `npx` 의존성을 없앴습니다.
- `NOTION_TOKEN`은 이후 환경 변수 방식으로 교체 가능하지만, 현재는 문제 해결을 위해 명시적으로 넣었습니다.

### 2. Notion MCP 서버 소스 준비 및 빌드
작업 디렉터리: `C:\Users\유신\.codex\notion-mcp-server`

```powershell
npm install
npm run build
```
- 빌드 후 `bin\cli.mjs`가 생성되어 MCP 서버 실행에 사용됩니다.
- `npm install` 시 이미 설치되어 있으면 `up to date` 메시지가 나타납니다.

### 3. MCP 서버 동작 확인
```powershell
node bin/cli.mjs --help
```
- 사용 방법이 출력되면 정상입니다.

### 4. Codex CLI 재시작
- Codex CLI를 완전히 종료 후 다시 실행하면 Notion MCP가 로컬 빌드된 서버를 통해 연결됩니다.

## 오류 해결 사례
- **오류 메시지:** `handshaking with MCP server failed: connection closed: initialize response`
- **원인:** `Authorization` 헤더에 `${NOTION_TOKEN}` 자리표시자가 그대로 전달되어 Notion API 인증이 실패.
- **조치:** `config.toml`의 `NOTION_TOKEN` 값을 실제 토큰 문자열로 설정하여 문제를 해결.

로그 예시 (`log\codex-tui.log`):
```
Authorization: Bearer ${NOTION_TOKEN}
```
→ 수정 후에는 실제 토큰이 치환되어 전송됩니다.

## 검증 방법
1. Codex CLI 실행 시 Notion MCP가 에러 없이 초기화되는지 확인합니다.
2. Notion 관련 MCP 도구(예: 검색) 호출 시 응답이 돌아오는지 테스트합니다.

## 추가 권장 사항
- 토큰 보안을 위해 추후 `setx NOTION_TOKEN <token>` 형태로 사용자 환경 변수에 저장하고 `config.toml`에서는 `${NOTION_TOKEN}`을 참조하도록 되돌리는 것을 추천합니다.
- `npm audit fix`를 실행해 보고된 취약점을 완화할 수 있습니다.
- 신규 버전이 발표되면 `npm update` 또는 레포지토리 갱신을 통해 최신 Notion MCP 서버를 유지하세요.
