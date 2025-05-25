import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

# Загрузка модели и названий классов из data.yaml
@st.cache_resource
def load_model_and_classes():
    model = YOLO('RTSD_YOLO11s_aug/yolo11s_aug/weights/best.pt')
    with open('data2.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        class_names = data['names']
    return model, class_names

def process_image(image, model, confidence_threshold, label_position, box_thickness, font_size, show_confidence):
    
    # Конвертируем PIL Image в numpy array (уже в RGB)
    img_np = np.array(image)
    
    # Копируем оригинальное изображение (без конвертации!)
    img_with_boxes = img_np.copy()
    
    # Предсказание
    results = model(img_np, conf=confidence_threshold)
    
    # img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    img_pil = Image.fromarray(img_with_boxes)  # Уже в RGB
    draw = ImageDraw.Draw(img_pil)
    temp_draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    detected_classes = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        label = model.model.names[class_id] if class_id in model.model.names else f"Класс {class_id}"
        if show_confidence:
            label += f" {confidence:.2f}"
        
        # Рисуем bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=box_thickness)
        
        # Получаем размеры текста
        text_bbox = temp_draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Определяем позицию текста
        if label_position == 'сверху':
            text_x = x1
            text_y = y1 - text_height - 5
        elif label_position == 'снизу':
            text_x = x1
            text_y = y2 + 5
        elif label_position == 'слева':
            text_x = max(0, x1 - text_width - 10)
            text_y = y1 + (y2 - y1)//2 - text_height//2
        elif label_position == 'справа':
            text_x = x2 + 10
            text_y = y1 + (y2 - y1)//2 - text_height//2
        else:  # внутри
            text_x = x1 + 5
            text_y = y1 + 5
        
        # Рисуем фон и текст
        draw.rectangle(
            [text_x-2, text_y-2, text_x + text_width+2, text_y + text_height+2],
            fill="red"
        )
        draw.text((text_x, text_y), label, fill="white", font=font)
        
        detected_classes.append((label, confidence))
    
    return np.array(img_pil), detected_classes

def main():
    st.title("Детекция дорожных знаков YOLOv11 (640)")
    
    # Отображение содержимого README.md
    with st.expander("ℹ️ О приложении (README)", expanded=False):
        try:
            with open("README_for_App.md", "r", encoding="utf-8") as f:
                readme_content = f.read()
            st.markdown(readme_content, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Не удалось загрузить README_for_App.md: {e}")
    
    # Настройки в сайдбаре
    st.sidebar.header("Настройки отображения")
    confidence_threshold = st.sidebar.slider(
        "Порог уверенности", 0.1, 1.0, 0.5, 0.01
    )
    show_confidence = st.sidebar.checkbox("Показывать уверенность", True)
    label_position = st.sidebar.radio(
        "Положение надписи",
        options=['сверху', 'снизу', 'слева', 'справа', 'внутри'],
        index=0
    )
    box_thickness = st.sidebar.slider("Толщина рамки", 1, 10, 3)
    font_size = st.sidebar.slider("Размер текста", 10, 50, 20)
    
    model, class_names = load_model_and_classes()
    
    # Мультизагрузка изображений
    # Для ограничения количества файлов:
    max_files = 10
    uploaded_files = st.file_uploader(
        f"Выберите до {max_files} изображений...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    if len(uploaded_files) > max_files:
        st.warning(f"Пожалуйста, загрузите не более {max_files} изображений")
        uploaded_files = uploaded_files[:max_files]

    if uploaded_files:
        # Обработка каждого изображения
        for i, uploaded_file in enumerate(uploaded_files):
            st.divider()
            st.subheader(f"Изображение {i+1}/{len(uploaded_files)}")
            
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Исходное изображение {i+1}", use_container_width=True)
            
            with st.spinner(f'Анализируем изображение {i+1}...'):
                result_img, detected_classes = process_image(
                    image, model, confidence_threshold, 
                    label_position, box_thickness, font_size, show_confidence
                )
                
                # Отображаем результат
                st.image(result_img, caption=f"Результат детекции {i+1}", use_container_width=True)
                
                # Выводим список обнаруженных знаков
                if detected_classes:
                    st.subheader(f"Обнаруженные дорожные знаки (изображение {i+1}):")
                    for label, conf in detected_classes:
                        st.write(f"- {label} (уверенность: {conf:.2f})")
                else:
                    st.warning(f"На изображении {i+1} дорожные знаки не обнаружены")

if __name__ == "__main__":
    main()
    
# Запуск приложения локально: streamlit run app.py