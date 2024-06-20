import os
import cv2
import face_recognition
import json
import rawpy
from imageio import imwrite
import numpy as np
import time

BASE_NAME = 'photos'
FACES_FILE = f"faces-{BASE_NAME}.json"
FACEDATA_FILE = f"face_data-{BASE_NAME}.json"
FACEENCONDINGCACHE_FILE = f"enconding-{BASE_NAME}.json"
DIR_PATH = f'./{BASE_NAME}'

filetypes = ('.png', '.webp', '.jpg', '.jpeg', '.gif', '.bmp', '.cr2', '.nef')

def convert_raw_to_jpg(raw_file, jpg_file):
  with rawpy.imread(raw_file) as raw:
    rgb = raw.postprocess()
    imwrite(jpg_file, rgb)

def recognize_faces_in_directory(directory, tolerance=0.6):
    # Carregar ou inicializar o cache
    if os.path.exists(FACEDATA_FILE):
        with open(FACEDATA_FILE, "r") as cache_file:
            cache = json.load(cache_file)
    else:
        cache = []
        
    if os.path.exists(FACEENCONDINGCACHE_FILE):
        with open(FACEENCONDINGCACHE_FILE, "r") as enconding_cache_file:
            face_encodings_list = json.load(enconding_cache_file)
    else:
        face_encodings_list = []

    knowned_faces = []
    face_data = []
    # face_encodings_list = []

    total_images = len([os.path.join(root, file_name) for root, _, files in os.walk(directory) for file_name in files if file_name.lower().endswith(filetypes)])
    processed_images = []

    for face in cache:
      face_encodings_list.extend(face['faces'])

    for root, dirs, files in os.walk(directory):
      for idx,file_name in enumerate(files):
        if any(face_encoding['image_path'] == os.path.join(root, file_name) for face_encoding in face_encodings_list):
          print('Pulando imagem já reconhecida')
          continue
        
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext in filetypes:
          image_path = os.path.join(root, file_name)
          print(f"Processando imagem {idx}/{total_images} - {image_path}")

          if file_ext in ('.cr2', '.nef'):
            # Converter RAW para JPEG
            jpg_path = os.path.splitext(image_path)[0] + '.jpg'
            convert_raw_to_jpg(image_path, jpg_path)
            image_path = jpg_path

          img = cv2.imread(image_path)
          rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          face_locations = face_recognition.face_locations(rgb_img)
          face_encodings = face_recognition.face_encodings(rgb_img, None, 1)
          
          if len(face_encodings) >= 1:
            print(f"Mais faces encontradas: {len(face_encodings)} - Total: {len(face_encodings_list) + len(face_encodings)}")
          else:
            print(f"Nenhuma face encontrada! - Total: {len(face_encodings_list) + len(face_encodings)}")
             
          processed_images.append(image_path)

          for i, face_encoding in enumerate(face_encodings):
            face_encodings_list.append({ "face_encoding": face_encoding.tolist(), "image_path": image_path, "face_id": i + 1, "face_location": face_locations[i] })
          
          with open(FACEENCONDINGCACHE_FILE, "w") as json_file:
            json.dump(face_encodings_list, json_file)

    for face_encoding1 in face_encodings_list:
      paths = []
      similar_faces = []
      faces_to_search = [face_encoding1['face_encoding']]
      clean_data = []

      for face_encoding2 in face_encodings_list:
        if(face_encoding2['image_path'] == face_encoding1['image_path']):
          if(face_encoding2['face_id'] == face_encoding1['face_id']):
            if any(path == face_encoding2['image_path'] for path in paths):
              continue
            similar_faces.append(face_encoding2)
            paths.append(face_encoding2['image_path'])
            continue

        results = face_recognition.compare_faces(np.array(faces_to_search), np.array(face_encoding2['face_encoding']), tolerance)
        for i, result in enumerate(results):
          if result:
              similar_faces.append(face_encoding2)
              faces_to_search.append(face_encoding2['face_encoding'])
              if all(path != face_encoding2['image_path'] for path in paths):
                paths.append(face_encoding2['image_path'])
              if all(data['path'] != face_encoding2['image_path'] for data in clean_data):
                clean_data.append({ 'path': face_encoding2['image_path'], 'location': face_encoding2['face_location']})
              
      face_data.append({
        "id": f'Face {len(face_data)+1}',
        "images": paths,
        "faces": similar_faces
      })
      knowned_faces.append({
        "id": f'Face {len(knowned_faces)+1}',
        "images": clean_data
      })
      with open(FACEDATA_FILE, "w") as json_file:
          json.dump(face_data, json_file)
      with open(FACES_FILE, "w") as json_file:
          json.dump(knowned_faces, json_file, indent=2)

if __name__ == "__main__":
    start_time = time.time()
    directory_path = DIR_PATH
    recognize_faces_in_directory(directory_path)
    end_time = time.time()

    delta = end_time - start_time
    
    # Extrai os componentes de horas, minutos e segundos da diferença
    segundos_totais = delta
    horas = int(segundos_totais // 3600)
    minutos = int((segundos_totais % 3600) // 60)
    segundos = int(segundos_totais % 60)
    
    print("Dados das faces reconhecidas e agrupadas salvos em face_data.json")
    print(f'{horas}:{minutos}:{segundos}')
