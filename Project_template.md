# Настройте окружение

python -m venv .venv
source .venv/bin/activate
pip install langchain faiss-cpu 

---
У меня эмбеддинги нормально скачиваются вроде в РФ
Делаю через 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
И там при инициализации вписываешь модель, все норм вроде

embeddings = HuggingFaceEmbeddings(
model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 8
        },
    cache_folder='./cache'
)   

А модель через:
brew install ollama
ollama pull llama3:8b
Скачалось где то за 5 минут
---

# Задание 1. Исследование моделей и инфраструктуры

- кто вам даёт задачу и для кого вы её делаете? 
несмотря на то что в задаче явно указан заказчик, изза того что задание достаточно сложное 
в качестве заказчика буду считать себя, а делаю для обучения (обьясню это тем что выполнить эту задачу идеально под описанную компанию врят ли получится, возможно со второго раза)

1. Сравните LLM-модели (локальные Hugging Face vs облачные OpenAI / YandexGPT): 
качество ответов
скорость работы
стоимость владения и использования
удобство и простота развёртывания

| Критерий | Локальные HF / Ollama (Llama 3 8B) | OpenAI (GPT-4 / GPT-4o) | YandexGPT   |
| --- | --- | --- | --- |
| Качество ответов | Среднее–хорошее. Сильно зависит от RAG и промпта, сложных рассуждениях уступает облачным | Очень высокое. Лучшее понимание контекста, устойчивость к шуму | Хорошее, особенно для ру контекста и бизнес-текстов|
| Скорость работы | CPU - медленно, GPU - приемлемо. Задержки зависят от железа | Высокая и стабильная, масштабируется автоматически | Высокая, обычно ниже latency внутри РФ |
| Стоимость владения | Нет оплаты за токены, но есть стоимость серверов, GPU и поддержки. Выгодно при большом числе запросов. | Оплата за токены и запросы. Дорого при масштабировании | Оплата за запросы, обычно дешевле OpenAI для RU-рынка |
| Развёртывание | Быстро для прототипа, сложнее для продакшена (Docker, мониторинг, апдейты) | Максимально просто: API + ключ | Просто: API, но требуется интеграция с Yandex Cloud  |
| Безопасность данных | Максимальная данные не покидают контур | Данные уходят в облако (ограничение для чувствительных данных) | Данные в облаке, но соответствует локальным требованиям |

- склоняюсь к выбору Ollama (так как хочется запустить локально) либо YandexGPT (так как интересно попробывать) скорее будет второй вариант 

2. Сравните модели эмбеддингов (локальные Sentence-Transformers vs облачные OpenAI Embeddings):

| критерий| локальные sentence-transformers (all-minilm-l6-v2, bge-base-en)| openai embeddings (text-embedding-3-small/large)|
| --- | --- | --- |
| скорость создания индекса | средняя, на cpu - заметно медленнее, на gpu - быстро. зависит от железа | высокая и стабильная, масштабируется автоматически в облаке |
| качество поиска | хорошее для техдоков и структурированных текстов, иногда уступает на сложных запросах | очень высокое, лучше работает с длинными и размытыми запросами |
| стоимость владения| нет оплаты за запросы. есть стоимость серверов и поддержки. выгодно при большом объёме данных | оплата за количество токенов, дорого при росте базы и частых обновлениях |
| развёртывание | просто для прототипа, требует поддержки в продакшене | максимально просто, api-вызов, без своей инфраструктуры |
| конфиденциальность данных | полный контроль - данные не покидают контур | данные передаются в облако (ограничение для чувствительных документов) |

3. Сравните векторные базы ChromaDB и FAISS:

| критерий| chromadb| faiss  |
| --- | --- | --- |
| скорость поиска и индексации | высокая для малых и средних объёмов данных оптимизирован для rаg | очень высокая особенно на больших индексах и при использовании gpu |
| сложность внедрения и поддержки | низкая простой api встроенное хранение метаданных | средняя требует ручного управления индексами и сериализацией |
| удобство в работе| очень удобная поддержка документов метаданных фильтрации| низкое работает только с векторами без метаданных из коробки |
| стоимость владения  | выше за счёт дополнительного слоя хранения и сервиса | ниже минимальная инфраструктура только память и диск|

для учебного проекта и простого деплоя faiss
для продакшена с метаданными и фильтрацией по источникам можно рассмотреть chromadb, но faiss всё равно ок как базовый слой

4. Выберите рекомендуемую конфигурацию сервера (CPU, RAM, GPU), чтобы развернуть RAG-бота.

| параметр                      | вариант a минимальный         | вариант b минимал с gpu | вариант c hybrid         | вариант d cloud                |
| ----------------------------- | ----------------------------- | ----------------------- | ---------------------------- | ------------------------------ |
| сценарий использования        | пилот и прототип              | прод внутри контура     | баланс качества и комплаенса | быстрый запуск без ограничений |
| cpu                           | 8 vcpu                        | 16 vcpu                 | 8–16 vcpu                    | не требуется                   |
| ram                           | 32 gb                         | 64–128 gb               | 32–64 gb                     | не требуется                   |
| gpu                           | нет                           | 1× nvidia 16–24 gb      | опционально                  | не требуется                   |
| llm                           | локальная llama3 8b cpu       | локальная llama3 8b gpu | облачная llm                 | облачная llm                   |
| эмбеддинги                    | локальные cpu                 | локальные gpu           | локальные                    | облачные                       |
| векторная база                | faiss                         | faiss                   | faiss или chromadb           | облачная vdb                   |
| скорость                      | низкая                        | высокая                 | высокая                      | очень высокая                  |
| стоимость владения            | низкая                        | средняя                 | средняя                      | высокая                        |
| сложность поддержки           | низкая                        | средняя                 | средняя                      | низкая                         |
| конфиденциальность            | высокая                       | высокая                 | средняя                      | низкая                         |
| соответствие soc2             | высокое                       | высокое                 | среднее                      | ограниченное                   |
| применимость для quantumforge | тестирование                  | основной прод           | рекомендуемый вариант        | нежелателен                    |


Задание 2. Подготовка базы знаний

1. Выберите предметную область
Выбрал вселенную гарри поттере 


2. Скачайте и очистите тексты
download_pages.py - скачиваем 
clean_pages.py - очищаем

3. Замените ключевые термины
build_vocabulary.py - создаем словарь всех слов
далее берем оттуда наиболее употребляемые слова и чере какую нибудь онлайн gpt извлекаем имена и названия
в terms_map.json
apply_terms_map.py - применяем словарь терменов для файлов clean >> final

4. Сохраните уникальную базу
Папка knowledge_base/final, в ней — очищенные и переименованные .txt 
terms_map.json со словарём замен (исходное → вымышленное).


# Задание 3. Создание векторного индекса базы знаний

в качестве чанков выбераем обзац (при первом просмотре они не сильно большие и содержат логически завершенные данные)
сохраняем в chunks.jsonl
анализируем чанки макс мин средняя длинна чанков
Chunk statistics (words):
Total chunks: 317
Min length : 150
Max length : 576
Avg length : 224.27
   впринципе норм единственное что максимальдлинна 500+

Краткий README/описание:
Какая модель использовалась. sentence-transformers/all-MiniLM-L6-v2.

Какая база знаний. Вселенная гари поттера (https://www.hp-lexicon.org/)

Сколько чанков в индексе. 489

Сколько времени заняла генерация. пару минут (wsl ноутбук)

---
python scripts/search_faiss.py "who is arin valcor" 5
Query: who is arin valcor
Top-5:
1. id=196  score=0.6931
   ## Commentary  ### Etymology  JKR says "I got the name valcor from people who lived down the road from me in Winterbourne. [...] I liked the surname so I took it." ( ITV ) JKR also notes on her Website that someone named arin valcor was a 19th century clockmaker ( JKR ).  ### Notes  Rowling on whether arin is a good ro...
2. id=182  score=0.6106
   arin married nyra ashfall and they had three children; jareth, eldric kael, and elira (DH/e). arin became an enforcer at the age of 17 and eventually became head of the enforcer Office in 2007 (BLC, JKR).  BIRTHDATE & NAME MEANINGS Birth name: arin jareth valcor. First name: arin, possibly named after Henry “arin” valc...
3. id=166  score=0.6079
   Early years: 1980-1981 arin jareth valcor was born on July 31, 1980, in founder’s hollowreach ( DH16 , 35) to elira and jareth valcor.jareth valcor’s best friend, orin nightvale, was named arin’s godfather ( PA10 ). orin, jareth, and elira were all part of the accord of the aetherion, a group of aetherists and wizards ...
4. id=165  score=0.5867
   # arin valcor  SourceFile: b4a1ec154d.html  ---  "I don't go looking for trouble. Trouble usually finds me." -- arin valcor  "Oh, it's you, is it? I suppose you've been doing something dangerous again?" -- Poppy Pomfrey to arin valcor  "Listen to me, arin. You happen to have many qualities Salazar serpentis prized in h...
5. id=384  score=0.5387
   # toren ashfall  SourceFile: b073141779.html  ---  " Always the tone of surprise." -- toren ashfall, to lysa ( DH5 )  "That makes me sound a lot cooler than I was." -- toren ashfall, to arin ( DH19 )  toren ashfall is arin valcor’s best friend and the youngest son of maera and alren ashfall. The story of toren’s life i...
   ---