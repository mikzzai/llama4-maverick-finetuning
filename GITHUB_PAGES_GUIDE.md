# Подробное руководство по публикации сайта на GitHub Pages

Это пошаговая инструкция по публикации сайта с руководством по дообучению Llama-4-Maverick на GitHub Pages.

## Содержание

- [Предварительные требования](#предварительные-требования)
- [Шаг 1: Создание репозитория на GitHub](#шаг-1-создание-репозитория-на-github)
- [Шаг 2: Настройка локального репозитория](#шаг-2-настройка-локального-репозитория)
- [Шаг 3: Настройка аутентификации](#шаг-3-настройка-аутентификации)
- [Шаг 4: Публикация кода на GitHub](#шаг-4-публикация-кода-на-github)
- [Шаг 5: Настройка GitHub Pages](#шаг-5-настройка-github-pages)
- [Шаг 6: Проверка публикации](#шаг-6-проверка-публикации)
- [Устранение проблем](#устранение-проблем)

## Предварительные требования

Перед началом убедитесь, что у вас:

1. Есть аккаунт на GitHub
2. Установлен Git на вашем компьютере
3. Базовое понимание работы с командной строкой

## Шаг 1: Создание репозитория на GitHub

1. Перейдите на [GitHub](https://github.com/) и войдите в свой аккаунт
2. Нажмите на кнопку "+" в правом верхнем углу и выберите "New repository"
3. Заполните форму создания репозитория:
   - Repository name: `llama4-maverick-finetuning` (или любое другое имя)
   - Description: `Руководство по дообучению Llama-4-Maverick в домашних условиях`
   - Visibility: Public (для GitHub Pages в бесплатном аккаунте)
   - Не инициализируйте репозиторий с README, .gitignore или лицензией
4. Нажмите "Create repository"

## Шаг 2: Настройка локального репозитория

Если у вас уже есть локальный репозиторий с проектом, пропустите этот шаг.

1. Создайте директорию для проекта:
   ```bash
   mkdir llama4-maverick-finetuning
   cd llama4-maverick-finetuning
   ```

2. Инициализируйте Git репозиторий:
   ```bash
   git init
   ```

3. Настройте данные пользователя для Git:
   ```bash
   git config user.name "Ваше Имя"
   git config user.email "ваш_email@example.com"
   ```

4. Создайте базовую структуру проекта:
   ```bash
   mkdir -p css js img
   ```

5. Создайте основные файлы (index.html, css/style.css, js/script.js, README.md)

6. Добавьте файлы в Git:
   ```bash
   git add .
   git commit -m "Initial commit"
   ```

## Шаг 3: Настройка аутентификации

GitHub больше не поддерживает аутентификацию по паролю через HTTPS. Вместо этого вам нужно использовать токен доступа (Personal Access Token) или SSH-ключи.

### Вариант 1: Использование Personal Access Token (PAT)

1. **Создайте токен доступа на GitHub**:
   - Перейдите в Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
   - Нажмите "Generate new token" -> "Generate new token (classic)"
   - Дайте токену название (например, "Llama-4-Maverick repo")
   - Выберите области действия (scopes): минимум нужен `repo` и `workflow`
   - Нажмите "Generate token"
   - **Важно**: Скопируйте токен сразу, так как вы больше не сможете его увидеть

2. **Настройте Git для использования токена**:
   ```bash
   git remote add origin https://USERNAME:YOUR_TOKEN@github.com/USERNAME/llama4-maverick-finetuning.git
   ```
   Замените `USERNAME` на ваше имя пользователя GitHub и `YOUR_TOKEN` на созданный токен.

### Вариант 2: Использование SSH-ключей (рекомендуется для постоянного использования)

1. **Проверьте наличие SSH-ключей**:
   ```bash
   ls -la ~/.ssh
   ```

2. **Если ключей нет, создайте их**:
   ```bash
   ssh-keygen -t ed25519 -C "ваш_email@example.com"
   ```
   (Просто нажимайте Enter на все вопросы для использования значений по умолчанию)

3. **Добавьте ключ в SSH-агент**:
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

4. **Скопируйте публичный ключ**:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   (Скопируйте весь вывод)

5. **Добавьте ключ на GitHub**:
   - Перейдите в Settings -> SSH and GPG keys -> New SSH key
   - Вставьте скопированный ключ и дайте ему название
   - Нажмите "Add SSH key"

6. **Настройте Git для использования SSH**:
   ```bash
   git remote add origin git@github.com:USERNAME/llama4-maverick-finetuning.git
   ```
   Замените `USERNAME` на ваше имя пользователя GitHub.

## Шаг 4: Публикация кода на GitHub

1. Убедитесь, что у вас есть файл `.github/workflows/pages.yml` для GitHub Actions:
   ```bash
   mkdir -p .github/workflows
   ```

2. Создайте файл `.github/workflows/pages.yml` со следующим содержимым:
   ```yaml
   name: Deploy to GitHub Pages

   on:
     push:
       branches: [ main ]
     workflow_dispatch:

   permissions:
     contents: read
     pages: write
     id-token: write

   concurrency:
     group: "pages"
     cancel-in-progress: true

   jobs:
     deploy:
       environment:
         name: github-pages
         url: ${{ steps.deployment.outputs.page_url }}
       runs-on: ubuntu-latest
       steps:
         - name: Checkout
           uses: actions/checkout@v3
         - name: Setup Pages
           uses: actions/configure-pages@v3
         - name: Upload artifact
           uses: actions/upload-pages-artifact@v1
           with:
             path: '.'
         - name: Deploy to GitHub Pages
           id: deployment
           uses: actions/deploy-pages@v1
   ```

3. Отправьте код в GitHub:
   ```bash
   git add .
   git commit -m "Add GitHub Pages workflow"
   git branch -M main
   git push -u origin main
   ```

## Шаг 5: Настройка GitHub Pages

1. Перейдите на страницу вашего репозитория на GitHub
2. Нажмите на вкладку "Settings"
3. В левом меню выберите "Pages"
4. В разделе "Build and deployment":
   - Source: выберите "GitHub Actions"
   - GitHub автоматически обнаружит workflow файл `.github/workflows/pages.yml`

5. Дождитесь завершения процесса сборки и публикации (обычно занимает несколько минут)

## Шаг 6: Проверка публикации

1. Вернитесь на вкладку "Actions" вашего репозитория
2. Проверьте статус последнего workflow запуска
3. Если workflow успешно завершился, ваш сайт будет доступен по адресу:
   ```
   https://USERNAME.github.io/llama4-maverick-finetuning/
   ```
   Замените `USERNAME` на ваше имя пользователя GitHub.

4. Перейдите по этому URL, чтобы убедиться, что ваш сайт успешно опубликован

## Устранение проблем

### Проблема: Ошибка аутентификации при push

**Симптом**:
```
remote: Support for password authentication was removed on August 13, 2021.
remote: Please see https://docs.github.com/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls for information on currently recommended modes of authentication.
fatal: Authentication failed for 'https://github.com/USERNAME/llama4-maverick-finetuning.git/'
```

**Решение**:
Используйте Personal Access Token или SSH-ключи, как описано в [Шаге 3](#шаг-3-настройка-аутентификации).

### Проблема: Сайт не публикуется

**Симптом**: Workflow завершается успешно, но сайт недоступен.

**Решение**:
1. Проверьте, что в репозитории есть файл `index.html` в корневой директории
2. Убедитесь, что в настройках Pages выбран правильный источник (GitHub Actions)
3. Проверьте вкладку "Actions" на наличие ошибок в workflow
4. Дождитесь завершения процесса публикации (может занять до 10 минут)

### Проблема: Ошибки в workflow

**Симптом**: Workflow завершается с ошибкой.

**Решение**:
1. Проверьте содержимое файла `.github/workflows/pages.yml`
2. Убедитесь, что у вас есть правильные разрешения для репозитория
3. Проверьте логи ошибок в разделе Actions

### Проблема: Изменения не отображаются на сайте

**Симптом**: Вы внесли изменения, но они не отображаются на опубликованном сайте.

**Решение**:
1. Убедитесь, что вы закоммитили и отправили изменения в ветку `main`
2. Проверьте, что workflow запустился и успешно завершился
3. Очистите кэш браузера или откройте сайт в режиме инкогнито

## Дополнительные ресурсы

- [Официальная документация GitHub Pages](https://docs.github.com/en/pages)
- [Документация по GitHub Actions](https://docs.github.com/en/actions)
- [Руководство по Git](https://git-scm.com/book/ru/v2)
