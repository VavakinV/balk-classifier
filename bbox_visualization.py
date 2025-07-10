import os
import cv2
import random
import matplotlib.pyplot as plt
import torch
from preprocessimage import preprocess_image
from torchvision.ops import box_iou

def visualize_test_predictions(model, test_data, target_size, device, n=5):
    """Визуализация предсказаний на тестовых данных с фактическими bbox"""
    model.eval()
    selected_samples = random.sample(test_data, min(n, len(test_data)))
    
    for sample in selected_samples:
        try:
            # Загрузка изображения
            img = cv2.imread(sample['img_path'])
            if img is None:
                print(f"Warning: Could not read image {sample['img_path']}")
                continue
                
            orig_h, orig_w = img.shape[:2]
            display_img = img.copy()
            
            # Предобработка изображения
            img_proc = preprocess_image(img, target_size)
            img_tensor = torch.FloatTensor(img_proc.transpose(2, 0, 1)).unsqueeze(0).to(device)
            
            # Получение предсказания
            with torch.no_grad():
                pred = model(img_tensor)
                pred_np = pred.cpu().numpy().squeeze()
            
            # Преобразование координат предсказания
            try:
                x_min_pred = int(pred_np[0] * orig_w)
                y_min_pred = int(pred_np[1] * orig_h)
                x_max_pred = int(pred_np[2] * orig_w)
                y_max_pred = int(pred_np[3] * orig_h)
            except Exception as e:
                print(f"Error converting coordinates for image {sample['img_path']}: {str(e)}")
                continue
            
            # Проверка валидности bounding box
            x_min_pred = max(0, min(orig_w, x_min_pred))
            y_min_pred = max(0, min(orig_h, y_min_pred))
            x_max_pred = max(0, min(orig_w, x_max_pred))
            y_max_pred = max(0, min(orig_h, y_max_pred))

            if x_min_pred >= x_max_pred or y_min_pred >= y_max_pred:
                print(f"Warning: Invalid predicted bbox coordinates for image {sample['img_path']}")
                continue
            
            # Фактический bounding box
            true_bbox = sample['bbox']
            x_min_true = int(true_bbox[0])
            y_min_true = int(true_bbox[1])
            x_max_true = int(true_bbox[2])
            y_max_true = int(true_bbox[3])
            
            # Расчет IoU для этого изображения
            pred_box = torch.tensor([[x_min_pred, y_min_pred, x_max_pred, y_max_pred]], dtype=torch.float)
            true_box = torch.tensor([[x_min_true, y_min_true, x_max_true, y_max_true]], dtype=torch.float)
            iou = box_iou(pred_box, true_box).item()
            
            # Отрисовка
            # 1. Фактический bbox (красный)
            cv2.rectangle(display_img, (x_min_true, y_min_true), (x_max_true, y_max_true), (0, 0, 255), 3)
            # 2. Предсказанный bbox (зеленый)
            cv2.rectangle(display_img, (x_min_pred, y_min_pred), (x_max_pred, y_max_pred), (0, 255, 0), 3)
            
            # Добавление текста с IoU
            cv2.putText(display_img, f"IoU: {iou:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Отображение
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(os.path.basename(sample['img_path']))
            plt.show()
            
        except Exception as e:
            print(f"Error processing image {sample['img_path']}: {str(e)}")