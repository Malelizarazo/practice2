from flask import Flask, render_template, Response
import cv2
import numpy as np
from collections import deque
import pyautogui
import time
import threading
import json
import os

app = Flask(__name__)

CONFIG_FILE = 'config.json'

# Cargar configuración si existe
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"Error al cargar configuración: {e}")
    return None

# Guardar configuración
def save_config(roi, capture_region):
    try:
        config = {
            'ROI': roi,
            'CAPTURE_REGION': capture_region
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print("Configuración guardada exitosamente")
    except Exception as e:
        print(f"Error al guardar configuración: {e}")

# Variables globales
jugadas = 0
ganadas = 0
jugadas_entre_ganadas = []
contador_entre = 0
frames_ganada = deque(maxlen=15)
UMBRAL_MOV = 50  # Umbral de movimiento moderado
COLOR_GANADA = 90
TIEMPO_QUIETUD_REQUERIDO = 9.0  # Segundos de quietud requeridos
tiempo_ultimo_movimiento = 0
puede_contar_movimiento = True  # Flag para controlar si podemos contar un nuevo movimiento

# Cargar configuración inicial
config = load_config()
if config:
    ROI = tuple(config['ROI'])
    CAPTURE_REGION = tuple(config['CAPTURE_REGION'])
else:
    # Valores por defecto actualizados según la configuración actual
    CAPTURE_REGION = (660, 440, 740, 700)  # (left, top, width, height)
    
    # Calcular ROI centrado dentro del CAPTURE_REGION
    roi_width = 200  # Ancho más pequeño para el ROI
    roi_height = 240  # Alto más pequeño para el ROI
    roi_x = 220  # Posición relativa dentro de la región de captura
    roi_y = 200  # Posición relativa dentro de la región de captura
    ROI = (roi_x, roi_y, roi_width, roi_height)  # ROI más pequeño y centrado

UMBRAL_COLORES_VICTORIA = 3  # Número máximo de colores distintos para considerar victoria

def draw_info(frame, estado, color_estado):
    # Dibujar ROI
    x, y, w, h = ROI
    cv2.rectangle(frame, (x, y), (x+w, y+h), color_estado, 2)
    
    # Dibujar información
    cv2.putText(frame, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_estado, 2)
    cv2.putText(frame, f"Jugadas: {jugadas}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Ganadas: {ganadas}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Jugadas desde última ganada: {contador_entre}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    # Dibujar zona de detección de ganada
    zona_y = y + int(h*0.6)
    cv2.line(frame, (x, zona_y), (x+w, zona_y), (0,255,255), 2)
    cv2.putText(frame, "Zona detección ganada", (x+w+5, zona_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

def show_debug_windows(frame, mask):
    x, y, w, h = ROI
    
    # Extraer y redimensionar ROI
    roi = frame[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, (200, 267))  # Hacer el ROI más grande para verlo mejor
    
    # Crear máscara en color para mejor visualización
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_resized = cv2.resize(mask_color, (200, 267))
    
    # Dibujar línea de detección en ROI ampliado
    zona_y = int(267 * 0.6)
    cv2.line(roi_resized, (0, zona_y), (200, zona_y), (0,255,255), 2)
    
    # Mostrar ventanas de debug
    cv2.imshow('ROI Ampliado', roi_resized)
    cv2.imshow('Mascara HSV', mask_resized)

def contar_colores_unicos(imagen):
    # Reducir la resolución de la imagen para hacer el conteo más eficiente
    reducida = cv2.resize(imagen, (50, 50))
    # Convertir a HSV que es más robusto para agrupar colores similares
    hsv = cv2.cvtColor(reducida, cv2.COLOR_BGR2HSV)
    # Cuantizar los colores para reducir variaciones pequeñas
    h = hsv[:,:,0] // 20  # Dividir el matiz en 9 grupos (180/20)
    s = hsv[:,:,1] // 51  # Dividir la saturación en 5 grupos (255/51)
    v = hsv[:,:,2] // 51  # Dividir el valor en 5 grupos
    # Combinar los canales para crear identificadores únicos de color
    colores = (h.astype(int) * 25 + s.astype(int) * 5 + v.astype(int))
    # Contar colores únicos
    return len(np.unique(colores))

def capture_and_show():
    global jugadas, ganadas, contador_entre, frames_ganada, ROI, CAPTURE_REGION, tiempo_ultimo_movimiento, puede_contar_movimiento
    frame_prev = None
    cv2.namedWindow('Captura', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Estado Detección', cv2.WINDOW_NORMAL)
    
    # Variables para ajuste
    roi_x, roi_y, roi_w, roi_h = ROI
    cap_x, cap_y, cap_w, cap_h = CAPTURE_REGION
    ajuste = 20
    ajuste_tamano = 10
    mensaje_temporal = None
    tiempo_mensaje = 0

    print("\nControles de ajuste:")
    print("ROI: IJKL para mover, +/- para tamaño")
    print("Captura: WASD para mover, Q/E para ancho, R/F para alto")
    print("P: Imprimir coordenadas actuales")
    print("S: Guardar configuración actual")
    print("Esc: Salir\n")

    while True:
        try:
            tiempo_actual = time.time()
            # Capturar pantalla
            screenshot = pyautogui.screenshot(region=(cap_x, cap_y, cap_w, cap_h))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if frame_prev is None:
                frame_prev = frame.copy()
                frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
                tiempo_ultimo_movimiento = tiempo_actual
                continue

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Verificar límites
            if roi_y+roi_h > frame.shape[0] or roi_x+roi_w > frame.shape[1]:
                print(f"ROI fuera de límites: frame shape={frame.shape}, ROI=({roi_x}, {roi_y}, {roi_w}, {roi_h})")
                continue

            # Extraer solo el ROI para mostrar
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
            
            # Detectar movimiento
            diff = cv2.absdiff(frame_prev_gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], frame_gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w])
            mov = np.sum(diff) / 255

            estado = "Esperando..."
            color_estado = (200, 200, 200)

            # Lógica de detección de movimiento con tiempo de quietud
            tiempo_sin_movimiento = tiempo_actual - tiempo_ultimo_movimiento
            
            if mov > UMBRAL_MOV:
                # Si hay movimiento, actualizamos el tiempo
                tiempo_ultimo_movimiento = tiempo_actual
                if puede_contar_movimiento:
                    # Solo contamos el movimiento si ha pasado suficiente tiempo de quietud
                    jugadas += 1
                    contador_entre += 1
                    estado = f"Jugada #{jugadas}"
                    color_estado = (255, 255, 0)
                    puede_contar_movimiento = False  # No contar más hasta que haya quietud
                else:
                    estado = "Movimiento detectado..."
                    color_estado = (200, 200, 0)
            elif tiempo_sin_movimiento >= TIEMPO_QUIETUD_REQUERIDO:
                # Si ha pasado suficiente tiempo sin movimiento, permitimos contar el siguiente
                puede_contar_movimiento = True
                estado = "Listo para siguiente movimiento"
                color_estado = (0, 200, 200)
            else:
                # Mostrar tiempo restante de quietud
                tiempo_restante = TIEMPO_QUIETUD_REQUERIDO - tiempo_sin_movimiento
                estado = f"Esperando quietud: {tiempo_restante:.1f}s"

            # Nueva lógica de detección de victoria
            num_colores = contar_colores_unicos(frame)
            frames_ganada.append(num_colores <= UMBRAL_COLORES_VICTORIA)

            if sum(frames_ganada) > 12:  # Si detecta pocos colores por varios frames
                ganadas += 1
                jugadas_entre_ganadas.append(contador_entre)
                estado = f">>> GANADA #{ganadas} (Colores: {num_colores})"
                color_estado = (0, 255, 0)
                contador_entre = 0
                frames_ganada.clear()

            frame_prev_gray = frame_gray.copy()

            # Crear una imagen más grande para mostrar la información
            display_height = 600
            display_width = 400
            display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            
            # Redimensionar ROI y colocarlo en la parte superior
            roi_display = cv2.resize(roi, (display_width, 400))
            display[0:400, 0:display_width] = roi_display
            
            # Dibujar información en la parte inferior con texto más grande
            cv2.putText(display, estado, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_estado, 2)
            cv2.putText(display, f"Jugadas: {jugadas}", (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
            cv2.putText(display, f"Ganadas: {ganadas}", (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            
            # Mostrar mensaje temporal si existe
            if mensaje_temporal and time.time() - tiempo_mensaje < 3:  # Mostrar mensaje por 3 segundos
                cv2.putText(display, mensaje_temporal, (10, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            else:
                mensaje_temporal = None
            
            # Dibujar línea de detección
            zona_y = int(400 * 0.6)
            cv2.line(display, (0, zona_y), (display_width, zona_y), (0,255,255), 2)
            cv2.putText(display, "Zona ganada", (10, zona_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            
            # Mostrar información de debug
            debug_info = f"Mov: {mov:.1f} Quietud: {tiempo_sin_movimiento:.1f}s"
            cv2.putText(display, debug_info, (10, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            # Crear una visualización del estado de detección
            estado_display = np.zeros((400, 400), dtype=np.uint8)
            if len(frames_ganada) > 0:
                # Mostrar el estado de detección como una barra de progreso
                altura_barra = int(400 * (sum(frames_ganada) / len(frames_ganada)))
                estado_display[400-altura_barra:400, :] = 255
            
            # Convertir a BGR para mostrar
            estado_display_color = cv2.cvtColor(estado_display, cv2.COLOR_GRAY2BGR)
            
            # Mostrar ventanas
            cv2.imshow('Captura', display)
            cv2.imshow('Estado Detección', estado_display_color)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc
                save_config(ROI, CAPTURE_REGION)
                mensaje_temporal = "Configuración guardada"
                tiempo_mensaje = time.time()
                time.sleep(1)  # Esperar para mostrar el mensaje
                break
            elif key == ord('p'):  # Imprimir coordenadas
                coordenadas = f"ROI=({roi_x},{roi_y},{roi_w},{roi_h}) CAPTURE=({cap_x},{cap_y},{cap_w},{cap_h})"
                print("\nCoordenadas actuales:")
                print(f"ROI = ({roi_x}, {roi_y}, {roi_w}, {roi_h})")
                print(f"CAPTURE_REGION = ({cap_x}, {cap_y}, {cap_w}, {cap_h})\n")
                mensaje_temporal = coordenadas
                tiempo_mensaje = time.time()
            elif key == ord('s'):  # Guardar configuración
                save_config(ROI, CAPTURE_REGION)
                mensaje_temporal = "Configuración guardada"
                tiempo_mensaje = time.time()
            # Ajustar ROI
            elif key == ord('i'):  # Arriba (usando IJKL para el ROI)
                roi_y = max(0, roi_y - ajuste)
            elif key == ord('k'):  # Abajo
                roi_y = min(frame.shape[0] - roi_h, roi_y + ajuste)
            elif key == ord('j'):  # Izquierda
                roi_x = max(0, roi_x - ajuste)
            elif key == ord('l'):  # Derecha
                roi_x = min(frame.shape[1] - roi_w, roi_x + ajuste)
            elif key == ord('='): # Aumentar tamaño ROI
                roi_w += ajuste_tamano
                roi_h += ajuste_tamano
            elif key == ord('-'):  # Reducir tamaño ROI
                roi_w = max(20, roi_w - ajuste_tamano)
                roi_h = max(30, roi_h - ajuste_tamano)
            # Ajustar región de captura
            elif key == ord('w'):  # Arriba
                cap_y -= ajuste
            elif key == ord('s'):  # Abajo
                cap_y += ajuste
            elif key == ord('a'):  # Izquierda
                cap_x -= ajuste
            elif key == ord('d'):  # Derecha
                cap_x += ajuste
            elif key == ord('q'):  # Reducir ancho
                cap_w = max(100, cap_w - ajuste)
            elif key == ord('e'):  # Aumentar ancho
                cap_w += ajuste
            elif key == ord('r'):  # Reducir alto
                cap_h = max(100, cap_h - ajuste)
            elif key == ord('f'):  # Aumentar alto
                cap_h += ajuste

            # Actualizar variables globales
            ROI = (roi_x, roi_y, roi_w, roi_h)
            CAPTURE_REGION = (cap_x, cap_y, cap_w, cap_h)
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error en captura_and_show: {str(e)}")
            time.sleep(0.1)

def gen_frames():
    global jugadas, ganadas, contador_entre, jugadas_entre_ganadas, frames_ganada
    frame_prev = None

    while True:
        try:
            screenshot = pyautogui.screenshot(region=CAPTURE_REGION)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if frame_prev is None:
                frame_prev = frame.copy()
                frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
                continue

            x, y, w, h = ROI
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if y+h > frame.shape[0] or x+w > frame.shape[1]:
                continue

            diff = cv2.absdiff(frame_prev_gray[y:y+h, x:x+w], frame_gray[y:y+h, x:x+w])
            mov = np.sum(diff) / 255

            estado = "Esperando..."
            color_estado = (200, 200, 200)

            if mov > UMBRAL_MOV:
                jugadas += 1
                contador_entre += 1
                estado = f"Jugada #{jugadas}"
                color_estado = (255, 255, 0)

            hsv = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
            zona = mask[int(h*0.6):, :]
            amarillo = np.sum(zona) / 255
            frames_ganada.append(amarillo > COLOR_GANADA)

            if sum(frames_ganada) > 12:
                ganadas += 1
                jugadas_entre_ganadas.append(contador_entre)
                estado = f">>> GANADA #{ganadas}"
                color_estado = (0, 255, 0)
                contador_entre = 0
                frames_ganada.clear()

            frame_prev_gray = frame_gray.copy()

            # Dibujar información en el frame
            draw_info(frame, estado, color_estado)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Error en la captura: {str(e)}")
            time.sleep(0.1)
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Iniciando aplicación...")
    capture_thread = threading.Thread(target=capture_and_show)
    capture_thread.daemon = True
    capture_thread.start()
    app.run(debug=True, use_reloader=False) 