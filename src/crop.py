from PIL import Image
import os

def read_coords(coords_path):
    """"Читает сохраненный текстовый файл после yolo_detect и преобразует
    в список, где каждый элемент - список нормализованных координат отдельного слова"""

    if os.path.exists(coords_path):
        try:
            with open(coords_path) as file:
                lst = file.readlines()
                lst_coords = []
                for line in lst:
                    try:
                        lst_coords.append([float(x) for x in line.strip().split()])
                    except ValueError:
                        print(f"Ошибка: Некорректные данные в строке: {line.strip()}")
                        continue

                return lst_coords

        except UnicodeDecodeError:
            print(f"Ошибка: Файл не является текстовым или использует неподдерживаемую кодировку: {coords_path}")
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")

    else:
        print("Ошибка: файл не найден")


def convert_to_pixel_coords(normalized_coords, image_width, image_height):
    """Преобразовывает каждый элемент списка с нормализованными координатами в пиксельные координаты"""

    pixel_coords = []
    for word in normalized_coords:
        normalized_x_center = word[1]
        normalized_y_center = word[2]
        normalized_width = word[3]
        normalized_height = word[4]

        x1 = int((normalized_x_center - normalized_width / 2) * image_width)
        y1 = int((normalized_y_center - normalized_height / 2) * image_height)
        x2 = int((normalized_x_center + normalized_width / 2) * image_width)
        y2 = int((normalized_y_center + normalized_height / 2) * image_height)

        pixel_coords.append((x1, y1, x2, y2))

    return pixel_coords


def calculate_center(bbox):
    """Вычисляет центр рамки в пиксельных координатах"""

    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    return [x_center, y_center]


def calculate_average_height(pixel_coords):
    """Вычисляет среднюю высоту bounding boxes"""
    heights = [y2 - y1 for x1, y1, x2, y2 in pixel_coords]
    return sum(heights) / len(heights) if heights else 0


def sort_coords(pixel_coords):
    """Сортирует bounding boxes слева-направо и сверху-вниз с адаптивным порогом для разделения строк."""

    if not pixel_coords:
        return []

    average_height = calculate_average_height(pixel_coords)

    y_threshold = max(10, average_height * 0.6)

    bboxes_with_center = [(bbox, calculate_center(bbox)) for bbox in pixel_coords]
    bboxes_with_center.sort(key=lambda item: item[1][1])
    lines = []
    current_line = []

    for bbox, center in bboxes_with_center:
        if not current_line:
            current_line.append((bbox, center))
        else:
            prev_center = current_line[-1][1]
            if abs(center[1] - prev_center[1]) <= y_threshold:
                current_line.append((bbox, center))
            else:
                lines.append(current_line)
                current_line = [(bbox, center)]

    if current_line:
        lines.append(current_line)

    sorted_coords = []
    for line in lines:
        line.sort(key=lambda item: item[1][0])
        sorted_coords.extend([bbox for bbox, _ in line])

    return sorted_coords


def crop_and_save_images(image_path, sorted_pixel_coords, output_dir):
    image_original = Image.open(image_path).convert("RGB")
    c = 0

    for i in range(len(sorted_pixel_coords)):
        cropped_image = image_original.crop(sorted_pixel_coords[i])
        c += 1
        save_path = os.path.join(output_dir, f'cropped_image{c}.jpg')
        cropped_image.save(save_path)
