// 노노그램 숫자 입력 → 이미지(테이블)로 변환 (기본 구조)
// 보드 만들기 버튼 클릭 시 행/열 입력 폼 생성
document.getElementById('make-board-btn').addEventListener('click', function() {
    const rows = parseInt(document.getElementById('board-rows').value, 10);
    const cols = parseInt(document.getElementById('board-cols').value, 10);
    if (isNaN(rows) || isNaN(cols) || rows < 1 || cols < 1) {
        alert('행과 열의 개수를 올바르게 입력하세요.');
        return;
    }
    const area = document.getElementById('numbers-input-area');
    area.innerHTML = '';
    // 행 입력 폼
    const rowDiv = document.createElement('div');
    rowDiv.innerHTML = `<b>각 행의 숫자 입력 (공백 또는 쉼표로 구분):</b><br>`;
    for (let i = 0; i < rows; i++) {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'row-number-input';
        input.placeholder = `행 ${i+1}`;
        rowDiv.appendChild(input);
        rowDiv.appendChild(document.createElement('br'));
    }
    area.appendChild(rowDiv);
    // 열 입력 폼
    const colDiv = document.createElement('div');
    colDiv.style.marginTop = '10px';
    colDiv.innerHTML = `<b>각 열의 숫자 입력 (공백 또는 쉼표로 구분):</b><br>`;
    for (let i = 0; i < cols; i++) {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'col-number-input';
        input.placeholder = `열 ${i+1}`;
        colDiv.appendChild(input);
        colDiv.appendChild(document.createElement('br'));
    }
    area.appendChild(colDiv);
    // 이미지 생성 버튼
    let genBtn = document.getElementById('generate-btn');
    if (!genBtn) {
        genBtn = document.createElement('button');
        genBtn.id = 'generate-btn';
        genBtn.textContent = '이미지 생성';
        area.appendChild(genBtn);
    } else {
        area.appendChild(genBtn);
    }
    document.getElementById('nonogram-output').innerHTML = '';
    // 이미지 생성 버튼 이벤트: 입력값으로 퍼즐 해를 찾아 테이블로 출력
    genBtn.onclick = function() {
        const rowInputs = Array.from(document.getElementsByClassName('row-number-input'));
        const colInputs = Array.from(document.getElementsByClassName('col-number-input'));
        const rowNums = rowInputs.map(input => parseLine(input.value));
        const colNums = colInputs.map(input => parseLine(input.value));
        // 입력값 검증
        if (!rowNums.every(isValidLine) || !colNums.every(isValidLine)) {
            document.getElementById('nonogram-output').innerHTML = '<p style="color:red">숫자 입력이 올바르지 않습니다.</p>';
            return;
        }
        // 퍼즐 해 찾기
        const solution = solveNonogram(rowNums, colNums);
        if (!solution) {
            document.getElementById('nonogram-output').innerHTML = '<p style="color:red">해답이 없습니다.</p>';
            return;
        }
        // 테이블로 출력
        document.getElementById('nonogram-output').innerHTML = renderNonogramTable(solution);
    };

    // 숫자 입력 파싱 (공백, 쉼표 모두 허용)
    function parseLine(str) {
        return str.split(/[,\s]+/).map(s => parseInt(s, 10)).filter(n => !isNaN(n) && n > 0);
    }
    function isValidLine(arr) {
        return Array.isArray(arr) && arr.every(n => Number.isInteger(n) && n > 0);
    }

    // 노노그램 퍼즐 해 찾기 (간단한 백트래킹, 첫 해만 반환)
    function solveNonogram(rowNums, colNums) {
        const rows = rowNums.length, cols = colNums.length;
        // 각 행에 대해 가능한 모든 패턴 미리 생성
        const rowPatterns = rowNums.map(nums => generateLinePatterns(nums, cols));
        let solution = null;
        function backtrack(r, board, colFill) {
            if (solution) return;
            if (r === rows) {
                // 열 조건 체크
                for (let c = 0; c < cols; c++) {
                    if (!checkLine(board.map(row => row[c]), colNums[c])) return;
                }
                solution = board.map(row => row.slice());
                return;
            }
            for (const pattern of rowPatterns[r]) {
                // colFill: 각 열에 지금까지 채운 칸 수
                let valid = true;
                for (let c = 0; c < cols; c++) {
                    if (pattern[c] === 1 && colFill[c] + 1 > (colNums[c].reduce((a,b)=>a+b,0) + colNums[c].length - 1)) {
                        valid = false; break;
                    }
                }
                if (!valid) continue;
                const newColFill = colFill.slice();
                for (let c = 0; c < cols; c++) if (pattern[c] === 1) newColFill[c]++;
                backtrack(r+1, [...board, pattern], newColFill);
            }
        }
        backtrack(0, [], Array(cols).fill(0));
        return solution;
    }

    // 한 줄(행/열)에서 가능한 모든 패턴 생성
    function generateLinePatterns(nums, len) {
        const res = [];
        function dfs(idx, arr, pos) {
            if (idx === nums.length) {
                if (arr.length === len) res.push(arr.slice());
                else if (arr.length < len) res.push(arr.concat(Array(len-arr.length).fill(0)));
                return;
            }
            let minSpace = idx === 0 ? 0 : 1;
            for (let i = pos; i + nums[idx] <= len; i++) {
                const next = arr.concat(Array(i-arr.length).fill(0), Array(nums[idx]).fill(1));
                if (next.length < len) next.push(0);
                dfs(idx+1, next, next.length);
            }
        }
        dfs(0, [], 0);
        return res;
    }

    // 한 줄이 조건에 맞는지 체크
    function checkLine(arr, nums) {
        const blocks = [];
        let cnt = 0;
        for (let v of arr) {
            if (v === 1) cnt++;
            else if (cnt > 0) { blocks.push(cnt); cnt = 0; }
        }
        if (cnt > 0) blocks.push(cnt);
        return JSON.stringify(blocks) === JSON.stringify(nums);
    }

    // 테이블 HTML 생성
    function renderNonogramTable(board) {
        let html = '<table class="nonogram-table">';
        for (const row of board) {
            html += '<tr>';
            for (const cell of row) {
                html += `<td${cell===1?' class="filled"':''}></td>`;
            }
            html += '</tr>';
        }
        html += '</table>';
        return html;
    }
});
