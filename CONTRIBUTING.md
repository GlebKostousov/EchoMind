# Contributing to Echo Mind

Спасибо за интерес к проекту Echo Mind! Мы приветствуем вклад от сообщества в любой форме.

## 🤝 Как помочь проекту

### Сообщить о проблеме
- Используйте [GitHub Issues](https://github.com/GlebKostousov/EchoMind/issues)
- Проверьте, что issue ещё не создан
- Предоставьте максимум информации: версия Python, OS, шаги воспроизведения

### Предложить улучшение
- Создайте issue с тегом `enhancement`
- Опишите use case и ожидаемое поведение
- Приложите примеры кода, если возможно

### Отправить Pull Request

1. Fork репозиторий
2. Создайте feature branch: `git checkout -b feature/amazing-feature`
3. Commit изменения: `git commit -m 'Add amazing feature'`
4. Push в branch: `git push origin feature/amazing-feature`
5. Откройте Pull Request

## 💻 Development Setup

```bash
# Клонировать fork
git clone https://github.com/YOUR_USERNAME/EchoMind.git
cd EchoMind

# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate

# Установить dependencies
pip install -e ".[dev]"

# Установить pre-commit hooks
pre-commit install
```

## ✅ Code Quality

### Pre-commit Hooks
Перед каждым commit автоматически запускаются:
- `ruff` — linting и форматирование
- `mypy` — type checking
- `pytest` — unit tests

### Запустить вручную
```bash
# Linting
ruff check src/

# Форматирование
ruff format src/

# Type checking
mypy src/

# Тесты
pytest tests/ -v --cov=echomind
```

## 📝 Code Style

- **PEP 8** через ruff
- **Type hints** обязательно
- **Docstrings** в Google style
- **Максимальная длина строки**: 100 символов

Пример:
```python
def embed_text(
    text: str,
    model: str = "default",
    normalize: bool = True,
) -> np.ndarray:
    """Создаёт embedding для текста.

    Args:
        text: Входной текст для embedding
        model: Название модели из HuggingFace Hub
        normalize: Нормализовать вектор в единичную длину

    Returns:
        NumPy array с embedding вектором

    Raises:
        ValueError: Если текст пустой
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    # ...
```

## 🧪 Testing

### Unit Tests
- Покрытие >80% для новых функций
- Моки для внешних зависимостей (API, БД)
- Используем `pytest` + `pytest-cov`

```python
# tests/unit/test_embeddings.py
def test_text_embedder_basic():
    embedder = TextEmbedder(model="test-model")
    result = embedder.embed("Hello world")
    assert result.shape == (768,)
    assert np.linalg.norm(result) == pytest.approx(1.0)
```

### Integration Tests
- Тестируют взаимодействие компонентов
- Используют тестовые БД (pytest fixtures)

### E2E Tests
- Полные сценарии использования
- Запускаются в Docker окружении

## 📚 Documentation

### Docstrings
Обязательны для:
- Публичных функций и методов
- Классов
- Модулей

### Markdown Docs
При добавлении новой функции:
1. Обновите `docs/api-reference.md`
2. Добавьте пример в `docs/examples/`
3. Обновите `README.md` если нужно

## 🌳 Git Workflow

### Branch Naming
- `feature/` — новые функции
- `fix/` — исправления багов
- `docs/` — документация
- `refactor/` — рефакторинг
- `test/` — добавление тестов

### Commit Messages
Следуем [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: добавлена поддержка видео embeddings
fix: исправлена утечка памяти в chunker
docs: обновлена документация API
test: добавлены тесты для reranker
```

### Pull Request Template
```markdown
## Описание
Краткое описание изменений

## Тип изменений
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Checklist
- [ ] Код следует style guide
- [ ] Добавлены/обновлены тесты
- [ ] Все тесты проходят
- [ ] Обновлена документация
- [ ] Changelog обновлён
```

## 🏗️ Архитектурные Принципы

1. **Модульность**: каждый компонент независим и тестируем
2. **Dependency Injection**: используем DI для flexibility
3. **Type Safety**: максимум type hints и mypy
4. **Избегаем vendor lock-in**: абстракции для LLM, vector DB
5. **Configuration over code**: настройки через YAML, не в коде

## 🔐 Security

- **Никогда не коммитьте**: API keys, tokens, credentials
- Используйте `.env` файлы (они в `.gitignore`)
- Сообщайте о security issues приватно: kostousjr@gmail.com

## 📜 License

Отправляя PR, вы соглашаетесь лицензировать свой код под Apache 2.0.

## 💬 Коммуникация

- **GitHub Issues**: баги и feature requests
- **Email**: kostousjr@gmail.com для приватных вопросов

---

**Спасибо за вклад в Echo Mind! 🚀**
