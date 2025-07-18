document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chat-box');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');

    // 다크/라이트 모드 토글 버튼 연결
    const modeToggle = document.getElementById('mode-toggle');
    let isDark = false;
    if (modeToggle) {
        modeToggle.addEventListener('click', function() {
            isDark = !isDark;
            document.body.classList.toggle('dark-mode', isDark);
            modeToggle.textContent = isDark ? '☀️ 라이트 모드' : '🌙 다크 모드';
            modeToggle.style.background = isDark ? '#2d2320' : '#fff6ed';
            modeToggle.style.color = isDark ? '#ffbe98' : '#ff7f50';
            modeToggle.style.borderColor = '#ffbe98';
        });
    }

    // 오늘의 명언 리스트 (원하면 추가/수정 가능)
    const quotes = [
        '성공은 준비와 기회의 만남이다. - 세네카',
        '포기하지 마라. 오늘의 어려움은 내일의 힘이 된다.',
        '노력은 배신하지 않는다.',
        '실패는 성공의 어머니이다.',
        '할 수 있다고 믿는 순간, 이미 반은 이룬 것이다. - 시어도어 루스벨트',
        '작은 성취가 큰 변화를 만든다.',
        '시작이 반이다.',
        '꿈을 꾸는 자만이 미래를 바꾼다.',
        '도전 없는 성공은 없다.',
        '오늘 걷지 않으면 내일은 뛰어야 한다.'
    ];
    // 오늘 날짜 기반으로 명언 선택
    const today = new Date();
    const quoteIdx = today.getFullYear() + today.getMonth() + today.getDate();
    const welcomeQuote = quotes[quoteIdx % quotes.length];

    // 최근 입력 10개를 저장할 배열
    const recentInputs = [];

    // 웰컴 메시지로 오늘의 명언 출력
    appendMessage('bot', `오늘의 명언: "${welcomeQuote}"`);

    // 랜덤 문구 표시용 div 생성
    let randomWordsDiv = document.getElementById('random-words');
    if (!randomWordsDiv) {
        randomWordsDiv = document.createElement('div');
        randomWordsDiv.id = 'random-words';
        document.body.appendChild(randomWordsDiv);
    }

    // 대화 내역을 저장할 배열
    const chatHistory = [
        { role: 'system', content: '당신은 친절한 한국어 챗봇입니다.' }
    ];

    function appendMessage(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ' + sender;
        msgDiv.textContent = text;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // 최근 입력 10개를 랜덤 위치에 표시
    function renderRandomWords() {
        randomWordsDiv.innerHTML = '';
        randomWordsDiv.style.position = 'fixed';
        randomWordsDiv.style.top = '0';
        randomWordsDiv.style.left = '0';
        randomWordsDiv.style.width = '100vw';
        randomWordsDiv.style.height = '100vh';
        randomWordsDiv.style.pointerEvents = 'none';
        randomWordsDiv.style.zIndex = '0';

        recentInputs.forEach(word => {
            const span = document.createElement('span');
            span.textContent = word;
            span.style.position = 'absolute';
            span.style.fontSize = (Math.random() * 0.7 + 1.1) + 'rem';
            span.style.opacity = '0.18';
            span.style.fontWeight = 'bold';
            span.style.color = '#3182f6';
            span.style.userSelect = 'none';
            // 랜덤 위치 (채팅창 영역을 피해서)
            const x = Math.random() * 80 + 5; // 5vw ~ 85vw
            const y = Math.random() * 80 + 5; // 5vh ~ 85vh
            span.style.left = x + 'vw';
            span.style.top = y + 'vh';
            randomWordsDiv.appendChild(span);
        });
    }

    async function botReply(userText) {
        appendMessage('bot', '...'); // 로딩 표시
        chatHistory.push({ role: 'user', content: userText });
        const apiKey = 'API키 자리'; // 반드시 본인 키로 교체
        const endpoint = 'https://api.openai.com/v1/chat/completions';
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`
                },
                body: JSON.stringify({
                    model: 'gpt-4o-mini',
                    messages: chatHistory,
                    max_tokens: 1000,
                    temperature: 0.7
                })
            });
            if (!response.ok) throw new Error('API 오류');
            const data = await response.json();햣
            // 기존 로딩 메시지 삭제
            const lastMsg = chatBox.querySelector('.message.bot:last-child');
            if (lastMsg && lastMsg.textContent === '...') chatBox.removeChild(lastMsg);
            const reply = data.choices?.[0]?.message?.content?.trim() || '답변을 가져올 수 없습니다.';
            appendMessage('bot', reply);
            chatHistory.push({ role: 'assistant', content: reply });
        } catch (err) {
            // 기존 로딩 메시지 삭제
            const lastMsg = chatBox.querySelector('.message.bot:last-child');
            if (lastMsg && lastMsg.textContent === '...') chatBox.removeChild(lastMsg);
            appendMessage('bot', 'API 호출 오류가 발생했습니다.');
        }
    }

    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const text = userInput.value.trim();
        if (text) {
            appendMessage('user', text);
            botReply(text);
            userInput.value = '';
            // 최근 입력 10개 관리 및 랜덤 표시
            if (!recentInputs.includes(text)) recentInputs.push(text);
            if (recentInputs.length > 10) recentInputs.shift();
            renderRandomWords();
        }
    });

    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });
});
