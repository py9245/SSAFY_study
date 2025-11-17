#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모의고사 HTML을 모바일 친화적으로 개선하고 정답 데이터를 주입하는 스크립트.
"""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, Tuple

BASE_DIR = Path(__file__).resolve().parent
VERSION_DIRS = [
    Path("모의ver1"),
    Path("모의ver2"),
    Path("모의ver3"),
]

NEW_STYLE = """
:root {
  --accent: #2c4a78;
  --correct: #1b8a5a;
  --incorrect: #d6455d;
  --card-bg: #ffffff;
  --card-shadow: 0 4px 18px rgba(18, 42, 88, 0.12);
  --card-shadow-strong: 0 6px 22px rgba(18, 42, 88, 0.18);
}
*,
*::before,
*::after {
  box-sizing: border-box;
}
body {
  font-family: "Segoe UI", "Nanum Gothic", "Apple SD Gothic Neo", sans-serif;
  margin: 0 auto;
  max-width: 960px;
  padding: 28px 18px 80px;
  background: #f5f7fb;
  color: #1f2d3d;
  line-height: 1.65;
}
header {
  text-align: center;
  margin-bottom: 32px;
  padding: 0 12px;
}
h1 {
  margin: 0 0 8px;
  font-size: clamp(1.6rem, 2.6vw + 1rem, 2.25rem);
  color: #1a2945;
}
.info {
  font-size: 0.95rem;
  color: #4f5d78;
  margin-bottom: 4px;
}
.question-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.question {
  background: var(--card-bg);
  border-radius: 18px;
  padding: 22px 24px;
  box-shadow: var(--card-shadow);
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.question.is-correct {
  box-shadow: 0 10px 26px rgba(27, 138, 90, 0.22);
}
.question.is-incorrect {
  box-shadow: 0 10px 26px rgba(214, 69, 93, 0.18);
}
.q-header {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
  align-items: baseline;
  margin-bottom: 14px;
}
.q-number {
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--accent);
}
.q-tags {
  font-size: 0.9rem;
  color: #5d6f92;
}
.prompt {
  margin-bottom: 16px;
  word-break: keep-all;
  white-space: pre-line;
}
.options {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 10px;
}
.options li {
  background: #f7f9fc;
  border-radius: 12px;
  padding: 10px 12px;
  transition: background 0.2s ease, transform 0.2s ease;
}
.options li:hover {
  background: #eef3fb;
  transform: translateY(-1px);
}
.options label {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  font-size: 0.95rem;
  color: #1f2d3d;
}
.options input[type="radio"] {
  accent-color: var(--accent);
  transform: scale(1.1);
}
textarea.answer-area {
  margin-top: 12px;
  border: 1px solid #c5d1e3;
  border-radius: 14px;
  background: #ffffff;
  width: 100%;
  padding: 12px 14px;
  font-size: 0.95rem;
  color: #1f2d3d;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
  min-height: 72px;
  font-family: inherit;
  resize: vertical;
}
textarea.answer-area.short {
  min-height: 64px;
}
textarea.answer-area.descriptive {
  min-height: 170px;
}
textarea.answer-area:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(44, 74, 120, 0.16);
  outline: none;
}
.action-row {
  margin-top: 14px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
.action-button {
  appearance: none;
  border: none;
  border-radius: 999px;
  padding: 10px 18px;
  background: var(--accent);
  color: #ffffff;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 4px 14px rgba(44, 74, 120, 0.26);
  transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.2s ease;
}
.action-button.secondary {
  background: #62759c;
}
.action-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(44, 74, 120, 0.3);
}
.action-button:active {
  transform: translateY(0);
  box-shadow: 0 4px 12px rgba(44, 74, 120, 0.22);
}
.feedback {
  margin-top: 16px;
  padding: 16px 18px;
  border-radius: 14px;
  background: rgba(44, 74, 120, 0.08);
  border-left: 4px solid rgba(44, 74, 120, 0.45);
  display: none;
  flex-direction: column;
  gap: 8px;
  font-size: 0.92rem;
}
.feedback.visible {
  display: flex;
}
.feedback.correct {
  background: rgba(27, 138, 90, 0.12);
  border-color: var(--correct);
  color: #1d6b46;
}
.feedback.incorrect {
  background: rgba(214, 69, 93, 0.12);
  border-color: var(--incorrect);
  color: #8f2f3d;
}
.feedback strong {
  font-weight: 700;
}
.feedback .answer-line {
  font-weight: 600;
}
.feedback .explanation-line {
  color: #2a374f;
}
footer {
  margin-top: 36px;
  text-align: center;
  font-size: 0.85rem;
  color: #6b7891;
}
@media (max-width: 900px) {
  body {
    padding: 24px 16px 72px;
  }
  .question {
    padding: 20px 18px;
  }
}
@media (max-width: 600px) {
  body {
    padding: 20px 14px 64px;
  }
  header {
    margin-bottom: 26px;
  }
  .q-header {
    flex-direction: column;
    align-items: flex-start;
  }
  .options {
    gap: 8px;
  }
  .options li {
    padding: 10px 12px;
  }
  .action-row {
    flex-direction: column;
    align-items: stretch;
  }
  .action-button {
    width: 100%;
    justify-content: center;
  }
}
@media (max-width: 420px) {
  body {
    padding: 18px 12px 56px;
  }
  .question {
    padding: 18px 14px;
    border-radius: 16px;
  }
  .options label {
    font-size: 0.92rem;
  }
  textarea.answer-area {
    font-size: 0.92rem;
  }
}
""".strip()


def find_answer_file(dir_path: Path) -> Path:
    for candidate in dir_path.glob("*.html"):
        stem_lower = candidate.stem.lower()
        if "answer" in stem_lower or "답" in candidate.stem:
            return candidate
    raise FileNotFoundError(f"정답 HTML 파일을 찾을 수 없습니다: {dir_path}")


def extract_exam_number(text_fragment: str) -> int:
    plain_text = unescape(re.sub(r"<.*?>", "", text_fragment))
    match = re.search(r"(\d+)", plain_text)
    if not match:
        raise ValueError(f"모의고사 번호를 찾을 수 없습니다: {plain_text}")
    return int(match.group(1))


def clean_text(html_fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", html_fragment, flags=re.I)
    fragment = re.sub(r"<.*?>", "", fragment)
    fragment = unescape(fragment)
    lines = [line.strip() for line in fragment.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_sectioned_answers(html_text: str) -> Dict[int, Dict[int, Dict[str, str]]]:
    sections: Dict[int, Dict[int, Dict[str, str]]] = {}
    for title, table_html in re.findall(
        r"<h2[^>]*>(.*?)</h2>\s*<table[^>]*>(.*?)</table>", html_text, re.S
    ):
        exam_no = extract_exam_number(title)
        rows = re.findall(r"<tr>(.*?)</tr>", table_html, re.S)
        answers: Dict[int, Dict[str, str]] = {}
        for row_html in rows[1:]:
            cells_html = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.S)
            if len(cells_html) < 4:
                continue
            number_text = clean_text(cells_html[0])
            number_match = re.search(r"\d+", number_text)
            if not number_match:
                continue
            question_no = int(number_match.group())
            type_text = clean_text(cells_html[1])
            answer_text = clean_text(cells_html[2])
            explanation_text = clean_text(cells_html[3])
            answers[question_no] = {
                "type": type_text,
                "answer": answer_text,
                "explanation": explanation_text,
            }
        sections[exam_no] = answers
    return sections


def split_answer_and_explanation(cell_html: str) -> Tuple[str, str]:
    parts = re.split(r"</span>", cell_html, maxsplit=1)
    remainder = parts[-1] if parts else cell_html
    remainder = remainder.lstrip()
    if "<br" in remainder:
        answer_part, explanation_part = remainder.split("<br", 1)
        if ">" in explanation_part:
            explanation_part = explanation_part.split(">", 1)[-1]
    else:
        answer_part, explanation_part = remainder, ""
    answer_text = clean_text(answer_part)
    explanation_text = clean_text(explanation_part)
    return answer_text, explanation_text


def parse_flat_table_answers(html_text: str) -> Dict[int, Dict[int, Dict[str, str]]]:
    table_match = re.search(r"<table[^>]*>(.*?)</table>", html_text, re.S)
    if not table_match:
        raise ValueError("정답 표를 찾지 못했습니다.")
    rows = re.findall(r"<tr>(.*?)</tr>", table_match.group(1), re.S)
    sections: Dict[int, Dict[int, Dict[str, str]]] = {}
    for row_html in rows[1:]:
        cells_html = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.S)
        if len(cells_html) < 5:
            continue
        exam_no = extract_exam_number(cells_html[0])
        number_text = clean_text(cells_html[1])
        if not number_text.isdigit():
            continue
        question_no = int(number_text)
        type_text = clean_text(cells_html[2])
        answer_text, explanation_text = split_answer_and_explanation(cells_html[4])
        entry = {
            "type": type_text,
            "answer": answer_text,
            "explanation": explanation_text,
        }
        sections.setdefault(exam_no, {})[question_no] = entry
    return sections


def parse_answers(answer_path: Path) -> Dict[int, Dict[int, Dict[str, str]]]:
    html_text = answer_path.read_text(encoding="utf-8")
    if "<h2" in html_text:
        return parse_sectioned_answers(html_text)
    return parse_flat_table_answers(html_text)


def ensure_meta_viewport(html_text: str) -> str:
    if 'name="viewport"' in html_text.lower():
        return html_text
    return re.sub(
        r"(<meta[^>]+charset[^>]*>\s*)",
        r'\1  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n',
        html_text,
        count=1,
    )


def replace_style_block(html_text: str) -> str:
    pattern = re.compile(r"<style[^>]*>.*?</style>", re.S)
    match = pattern.search(html_text)
    if not match:
        return html_text
    replacement = f"<style>\n{NEW_STYLE}\n</style>"
    return html_text[: match.start()] + replacement + html_text[match.end() :]


def strip_existing_script_snippets(html_text: str) -> str:
    html_text = re.sub(
        r"\s*<script>\s*window\.EXAM_ANSWER_DATA\s*=\s*.*?</script>",
        "",
        html_text,
        flags=re.S,
    )
    html_text = re.sub(
        r"\s*<script[^>]+exam-interactive\.js[^>]*></script>",
        "",
        html_text,
        flags=re.S,
    )
    return html_text


def inject_data_script(html_text: str, answer_payload: Dict[int, Dict[str, str]]) -> str:
    data_json = json.dumps(answer_payload, ensure_ascii=False, indent=2)
    script_block = (
        "<script>\n"
        "  window.EXAM_ANSWER_DATA = "
        + data_json
        + ";\n"
        "</script>\n"
        '<script src="../exam-interactive.js"></script>\n'
    )
    return re.sub(
        r"</body>\s*</html>\s*$",
        script_block + "</body>\n</html>",
        html_text,
        flags=re.S,
    )


def update_exam_file(exam_path: Path, answers: Dict[int, Dict[str, str]]) -> None:
    html_text = exam_path.read_text(encoding="utf-8")
    html_text = strip_existing_script_snippets(html_text)
    html_text = ensure_meta_viewport(html_text)
    html_text = replace_style_block(html_text)
    html_text = inject_data_script(html_text, answers)
    exam_path.write_text(html_text, encoding="utf-8")


def iter_exam_files(dir_path: Path, answer_file: Path) -> Iterable[Path]:
    for html_file in sorted(dir_path.glob("*.html")):
        if html_file == answer_file:
            continue
        stem_lower = html_file.stem.lower()
        if "answer" in stem_lower or "정답" in stem_lower:
            continue
        yield html_file


def main() -> None:
    for relative_dir in VERSION_DIRS:
        dir_path = BASE_DIR / relative_dir
        if not dir_path.exists():
            continue
        answer_file = find_answer_file(dir_path)
        answer_map = parse_answers(answer_file)
        for exam_file in iter_exam_files(dir_path, answer_file):
            exam_no_match = re.search(r"(\d+)", exam_file.stem)
            if not exam_no_match:
                print(f"[SKIP] 번호를 찾을 수 없어 건너뜁니다: {exam_file}")
                continue
            exam_no = int(exam_no_match.group(1))
            exam_answers = answer_map.get(exam_no)
            if not exam_answers:
                print(f"[WARN] 정답 데이터 없음: {exam_file}")
                continue
            update_exam_file(exam_file, exam_answers)
            print(f"[OK] {exam_file.relative_to(BASE_DIR)} 갱신 완료")


if __name__ == "__main__":
    main()
