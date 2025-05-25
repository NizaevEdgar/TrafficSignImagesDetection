# Детекция дорожных знаков и определение их категорий

Система распознавания дорожных знаков с использованием сверточных нейросетей и датасета RTSD (Russian Traffic Sign Dataset). Проект предназначен для экспериментов с классификацией и детекцией дорожных знаков, актуален для систем помощи водителю и беспилотного транспорта.

---

## 🗂️ Исходные данные

- **Исходный датасет RTSD (Russian Traffic Sign Dataset)**  
   🌐[Ссылка на датасет на Kaggle](https://www.kaggle.com/datasets/watchman/rtsd-dataset) В данном датасете имеются размеченные дорожные знаки с российских дорог общего пользования.

- 📄 **Категории**: Знаки были сгруппированы по 8 основным группам согласно классификации ПДД РФ:
1. запрещающие
2. предупреждающие
3. знаки приоритета
4. предписывающие
5. особые предписания
6. информационные
7. сервисные
8. с дополнительной информацией

---

## 🧾 Структура проекта

- Файл [*Zala_Project_1.ipynb*](https://github.com/NizaevEdgar/TrafficSignImagesDetection/blob/main/Zala_Project_1.ipynb) - Загрузка исходного датасета, просмотр исходных данных, группировка по 8 категориям.
- Файл [*Zala_Project_2.ipynb*](https://github.com/NizaevEdgar/TrafficSignImagesDetection/blob/main/Zala_Project_2.ipynb) - Подготовка и обучение модели YOLO11s на исходных данных, без аугментации.
- Файл [*Zala_Project_3.ipynb*](https://github.com/NizaevEdgar/TrafficSignImagesDetection/blob/main/Zala_Project_3.ipynb) - Борьба с дисбалансом в категориях, использование аугментации, подготовка и обучение модели YOLO11s на аугментированных данных.
- Файл [*Zala_Project_3-1.ipynb*](https://github.com/NizaevEdgar/TrafficSignImagesDetection/blob/main/Zala_Project_3-1.ipynb) - Экспериментальный файл, подготовка и обучение модели YOLO11s на аугментированных данных с размером изображения imgsz=1280.

---
## 📊 Результаты

| Модель                              | mAP@50 | mAP@50:95 | Precision | Recall | F1-score | Скорость (ms/img) |
|-------------------------------------|--------|-----------|-----------|--------|----------|--------------------|
| YOLO11s без аугментации (imgsz=640) | 0.9196 | 0.6816    | 0.8797    | 0.8724 | 0.8760   | 3.40               |
| YOLO11s с аугментацией (imgsz=640)  | 0.9398 | 0.7082    | 0.8917    | 0.8916 | 0.8916   | 3.67               |
| YOLO11s с аугментацией (imgsz=1280) | 0.9479 | 0.7200    | 0.8956    | 0.9015 | 0.8985   | 13.76              |

---
## 🚀 Демо-приложение

Это приложение демонстрирует модель YOLO для детекции дорожных знаков. Используется модель из файла [*Zala_Project_3.ipynb*](https://github.com/NizaevEdgar/TrafficSignImagesDetection/blob/main/Zala_Project_3.ipynb)

[Открыть демо на Hugging Face](https://huggingface.co/spaces/NizaevEdgar/TrafficSignImagesDetection)

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/NizaevEdgar/TrafficSignImagesDetection)

## 💡 Планы по доработке

- Проверка начальных данных (было выявлено, что не все изначальные метки поставлены корректно)
- Использование новых аугментаций, увеличение количества эпох обучения и уменьшение learning rate.
- Добавление GAN (Generative Adversarial Network) для генерации синтетических изображений дорожных знаков для решения проблемы дисбаланса данных в категориях.
- Использование других моделей для обучения (RetinaNet + Focal Loss, Faster R-CNN)
- Дообучить ранее обученную модель YOLO на новых данных и проверить разницу.
