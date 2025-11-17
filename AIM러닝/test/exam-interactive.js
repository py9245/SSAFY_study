(() => {
  const answerData = window.EXAM_ANSWER_DATA || {};

  const normalize = (value) =>
    (value ?? '')
      .toString()
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();

  const escapeHtml = (value) =>
    (value ?? '')
      .toString()
      .replace(/[&<>"']/g, (char) => {
        switch (char) {
          case '&':
            return '&amp;';
          case '<':
            return '&lt;';
          case '>':
            return '&gt;';
          case '"':
            return '&quot;';
          case "'":
            return '&#39;';
          default:
            return char;
        }
      });

  const formatContent = (value) => escapeHtml(value).replace(/\n/g, '<br>');

  const toKey = (value) =>
    (value ?? '')
      .toString()
      .replace(/\s+/g, '')
      .toLowerCase();

  const ensureFeedbackEl = (questionEl) => {
    let feedback = questionEl.querySelector('.feedback');
    if (!feedback) {
      feedback = document.createElement('div');
      feedback.className = 'feedback';
      feedback.innerHTML = `
        <div class="feedback-message"></div>
        <div class="feedback-answer answer-line"></div>
        <div class="feedback-explanation explanation-line"></div>
      `;
      questionEl.appendChild(feedback);
    }
    return feedback;
  };

  const showFeedback = (
    questionEl,
    status,
    message,
    answerText,
    explanationText
  ) => {
    const feedback = ensureFeedbackEl(questionEl);
    feedback.classList.add('visible');
    feedback.classList.remove('correct', 'incorrect');
    questionEl.classList.remove('is-correct', 'is-incorrect');

    if (status === true) {
      feedback.classList.add('correct');
      questionEl.classList.add('is-correct');
    } else if (status === false) {
      feedback.classList.add('incorrect');
      questionEl.classList.add('is-incorrect');
    }

    const messageEl = feedback.querySelector('.feedback-message');
    const answerEl = feedback.querySelector('.feedback-answer');
    const explanationEl = feedback.querySelector('.feedback-explanation');

    if (messageEl) {
      messageEl.textContent = message || '';
    }
    if (answerEl) {
      answerEl.innerHTML = answerText
        ? `<strong>정답:</strong> ${formatContent(answerText)}`
        : '';
    }
    if (explanationEl) {
      explanationEl.innerHTML = explanationText
        ? `<strong>해설:</strong> ${formatContent(explanationText)}`
        : '';
    }
  };

  const createButton = (label, variant) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `action-button${variant ? ` ${variant}` : ''}`;
    button.textContent = label;
    return button;
  };

  const attachActionRow = (questionEl) => {
    let actionRow = questionEl.querySelector('.action-row');
    if (!actionRow) {
      actionRow = document.createElement('div');
      actionRow.className = 'action-row';
      questionEl.appendChild(actionRow);
    }
    return actionRow;
  };

  const extractOptionValue = (optionEl, index) => {
    if (optionEl.dataset.optionValue) {
      return optionEl.dataset.optionValue;
    }

    const strongText =
      optionEl.querySelector('strong')?.textContent?.trim() ?? '';
    const textContent = (optionEl.textContent || '').trim();
    const candidates = [];

    if (strongText) {
      candidates.push(strongText);
    }
    candidates.push(textContent);

    for (const candidate of candidates) {
      const match = candidate.match(/[A-Za-z가-힣0-9]+/);
      if (match && match[0]) {
        optionEl.dataset.optionValue = match[0];
        return match[0];
      }
    }

    const fallback = String.fromCharCode(65 + (index % 26));
    optionEl.dataset.optionValue = fallback;
    return fallback;
  };

  const ensureMultipleChoiceInputs = (questionEl, key) => {
    const existingRadios = questionEl.querySelectorAll('input[type="radio"]');
    if (existingRadios.length) {
      return Array.from(existingRadios);
    }

    const options = questionEl.querySelectorAll('.options li');
    if (!options.length) {
      return [];
    }

    const groupName = `q-${key}`;

    options.forEach((optionEl, index) => {
      if (optionEl.querySelector('input[type="radio"]')) {
        return;
      }

      const optionValue = extractOptionValue(optionEl, index);
      const label = document.createElement('label');
      label.className = 'option-label';

      const input = document.createElement('input');
      input.type = 'radio';
      input.name = groupName;
      input.value = optionValue;

      label.appendChild(input);

      const fragment = document.createDocumentFragment();
      while (optionEl.firstChild) {
        fragment.appendChild(optionEl.firstChild);
      }
      label.appendChild(fragment);
      optionEl.appendChild(label);
    });

    return Array.from(questionEl.querySelectorAll('input[type="radio"]'));
  };

  const ensureShortAnswerArea = (questionEl) => {
    let textarea = questionEl.querySelector('textarea.answer-area');
    if (!textarea) {
      textarea = document.createElement('textarea');
      textarea.className = 'answer-area short';
      textarea.placeholder = '답안을 입력하세요.';

      const anchor =
        questionEl.querySelector('.note, .action-row, .feedback') ?? null;
      if (anchor) {
        questionEl.insertBefore(textarea, anchor);
      } else {
        questionEl.appendChild(textarea);
      }
    }
    return textarea;
  };

  const handleMultipleChoice = (questionEl, qData, key) => {
    if (questionEl.dataset.multipleChoiceReady === 'true') {
      return;
    }
    const radios = ensureMultipleChoiceInputs(questionEl, key);
    if (!radios.length) {
      return;
    }

    questionEl.dataset.multipleChoiceReady = 'true';

    const optionItems = radios
      .map((radio) => radio.closest('li'))
      .filter(Boolean);

    const markSelection = (activeRadio) => {
      optionItems.forEach((item) => item.classList?.remove('selected-option'));
      const owner = activeRadio.closest('li');
      if (owner) {
        owner.classList.add('selected-option');
      }
    };

    radios.forEach((radio) => {
      radio.addEventListener('change', () => {
        if (!radio.checked) {
          return;
        }
        markSelection(radio);
        const selectedValue = radio.value;
        const isCorrect =
          normalize(selectedValue) === normalize(qData.answer);
        const message = isCorrect
          ? '정답입니다! 잘하셨어요.'
          : `오답입니다. 선택한 답: ${selectedValue}`;
        showFeedback(
          questionEl,
          isCorrect,
          message,
          qData.answer,
          qData.explanation
        );
      });
    });
  };

  const handleShortAnswer = (questionEl, qData) => {
    if (questionEl.dataset.shortAnswerReady === 'true') {
      return;
    }

    const textarea = ensureShortAnswerArea(questionEl);
    if (!textarea) {
      return;
    }

    const actionRow = attachActionRow(questionEl);
    let checkButton = actionRow.querySelector(
      '[data-action="check-short-answer"]'
    );
    if (!checkButton) {
      checkButton = createButton('채점하기', '');
      checkButton.dataset.action = 'check-short-answer';
      actionRow.appendChild(checkButton);
    }

    const evaluate = () => {
      const userInput = textarea.value;
      if (!userInput || !userInput.trim()) {
        showFeedback(
          questionEl,
          null,
          '먼저 답안을 입력해주세요.',
          '',
          ''
        );
        return;
      }
      const isCorrect = normalize(userInput) === normalize(qData.answer);
      const message = isCorrect
        ? '정답입니다!'
        : `오답입니다. 정답 제시: ${qData.answer}`;
      showFeedback(
        questionEl,
        isCorrect,
        message,
        qData.answer,
        qData.explanation
      );
    };

    checkButton.addEventListener('click', evaluate);
    textarea.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        evaluate();
      }
    });

    questionEl.dataset.shortAnswerReady = 'true';
  };

  const handleDescriptive = (questionEl, qData) => {
    if (questionEl.dataset.descriptiveReady === 'true') {
      return;
    }
    const actionRow = attachActionRow(questionEl);
    let button = actionRow.querySelector('[data-action="show-descriptive"]');
    if (!button) {
      button = createButton('해설 보기', 'secondary');
      button.dataset.action = 'show-descriptive';
      actionRow.appendChild(button);
    }

    button.addEventListener('click', () => {
      showFeedback(
        questionEl,
        null,
        '모범 답안과 해설을 확인하세요.',
        qData.answer,
        qData.explanation
      );
    });

    questionEl.dataset.descriptiveReady = 'true';
  };

  const findAnswerEntry = (rawKey) => {
    if (!rawKey) {
      return null;
    }
    const variants = [];
    const trimmed = rawKey.toString().trim();
    if (trimmed) {
      variants.push(trimmed);
      variants.push(trimmed.replace(/^0+/, ''));
    }
    const numeric = Number(trimmed);
    if (!Number.isNaN(numeric)) {
      variants.push(String(numeric));
    }

    const seen = new Set();
    for (const variant of variants) {
      const key = variant || '';
      if (!key || seen.has(key)) {
        continue;
      }
      seen.add(key);
      if (Object.prototype.hasOwnProperty.call(answerData, key)) {
        return { key, data: answerData[key] };
      }
    }
    return null;
  };

  const resolveQuestionData = (questionEl) => {
    const extractors = [
      () => questionEl.dataset.questionKey,
      () => {
        const numberNode = questionEl.querySelector('.q-number');
        if (numberNode?.textContent) {
          const match = numberNode.textContent.match(/\d+/);
          if (match) {
            return match[0];
          }
        }
        return null;
      },
      () => {
        if (questionEl.id) {
          const match = questionEl.id.match(/\d+/);
          if (match) {
            return match[0];
          }
        }
        return null;
      },
      () => {
        const metaSpans = questionEl.querySelectorAll('.question-meta span');
        for (const span of metaSpans) {
          if (span?.textContent) {
            const match = span.textContent.match(/\d+/);
            if (match) {
              return match[0];
            }
          }
        }
        return null;
      },
      () => {
        const headerMatch = (questionEl.textContent || '').match(/문제\s*(\d+)/);
        return headerMatch ? headerMatch[1] : null;
      },
    ];

    for (const extractor of extractors) {
      const rawKey = extractor();
      const entry = findAnswerEntry(rawKey);
      if (entry) {
        questionEl.dataset.questionKey = entry.key;
        return entry;
      }
    }
    return null;
  };

  const determineQuestionKind = (questionEl, qData) => {
    const normalizedType = toKey(qData.type || '');
    if (
      normalizedType.includes('서술') ||
      normalizedType.includes('논술') ||
      normalizedType.includes('기술') ||
      normalizedType.includes('essay')
    ) {
      return 'descriptive';
    }

    const hasChoiceInputs = questionEl.querySelector('input[type="radio"]');
    const hasChoiceList = questionEl.querySelector('.options li');
    if (
      hasChoiceInputs ||
      hasChoiceList ||
      normalizedType.includes('객관') ||
      normalizedType.includes('선택')
    ) {
      return 'choice';
    }

    if (questionEl.querySelector('textarea.answer-area')) {
      return 'short';
    }

    if (
      normalizedType.includes('주관') ||
      normalizedType.includes('단답') ||
      normalizedType.includes('숫자') ||
      normalizedType.includes('계산') ||
      normalizedType.includes('short') ||
      normalizedType.includes('기입')
    ) {
      return 'short';
    }

    return 'descriptive';
  };

  const init = () => {
    if (!answerData || !Object.keys(answerData).length) {
      return;
    }

    document.querySelectorAll('.question').forEach((questionEl) => {
      const resolved = resolveQuestionData(questionEl);
      if (!resolved) {
        return;
      }
      const { key, data } = resolved;
      const kind = determineQuestionKind(questionEl, data);

      if (kind === 'choice') {
        handleMultipleChoice(questionEl, data, key);
        return;
      }

      if (kind === 'short') {
        handleShortAnswer(questionEl, data);
        return;
      }

      handleDescriptive(questionEl, data);
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
