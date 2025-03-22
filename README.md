# Key Master Detector

Aplicación de detección y seguimiento de jugadas para Key Master, utilizando procesamiento de imágenes en tiempo real.

## Características

- Detección automática de jugadas basada en movimiento
- Contador de jugadas y victorias
- Interfaz gráfica con visualización en tiempo real
- Configuración ajustable de regiones de captura
- Guardado automático de configuración

## Requisitos

- Python 3.8+
- OpenCV
- Flask
- NumPy
- PyAutoGUI

## Instalación

1. Clonar el repositorio

```bash
git clone [URL_DEL_REPOSITORIO]
cd key-master-detector
```

2. Crear un entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

1. Ejecutar la aplicación:

```bash
python app.py
```

2. Controles:

- ROI (Region of Interest):
  - I/J/K/L: Mover arriba/izquierda/abajo/derecha
  - +/-: Aumentar/reducir tamaño
- Región de Captura:
  - W/A/S/D: Mover arriba/izquierda/abajo/derecha
  - Q/E: Ajustar ancho
  - R/F: Ajustar alto
- Otros:
  - P: Mostrar coordenadas actuales
  - S: Guardar configuración
  - Esc: Salir

## Configuración

La aplicación guarda la configuración en `config.json`, que incluye:

- Coordenadas del ROI
- Coordenadas de la región de captura

## Detección de Jugadas

- Una jugada se cuenta cuando:
  1. Se detecta movimiento significativo en el fondo
  2. El fondo permanece quieto por 10 segundos
- Una victoria se registra cuando la llave aparece en la zona de detección
