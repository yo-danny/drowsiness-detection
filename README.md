# EN
😴 ## Drowsiness Detector

This project implements a real-time drowsiness detection system using computer vision. It uses a webcam to monitor the user's eyes and detect signs of fatigue based on the Eye Aspect Ratio (EAR) metric.

When the system detects that the eyes remain closed for a defined period, an audible alert is triggered.

🚀 ## Features


Real-time face detection
Eye Aspect Ratio (EAR) calculation
Closed-eye detection
Automatic sound alert for drowsiness
Visualization of eye landmarks
Real-time EAR value display


🧠 ##How it works

The system uses facial landmarks to identify key points around the eyes.

The EAR is calculated using the formula:

EAR = (||P2 - P6|| + ||P3 - P5||) / (2 * ||P1 - P4||)
Lower EAR values indicate closed eyes
If EAR stays below a threshold for a continuous time → drowsiness detected


🛠️ ## Technologies Used
Python
OpenCV
MediaPipe
Chime (for audio alerts)


📄 ## License

This project is open-source and available for educational and experimental use.


# PT 
😴 ## Drowsiness Detector (Detector de Sonolência)

Este projeto implementa um detector de sonolência em tempo real utilizando visão computacional. Ele usa a webcam para monitorar os olhos do usuário e detectar sinais de cansaço com base na métrica EAR (Eye Aspect Ratio).

Quando o sistema identifica que os olhos permanecem fechados por um período definido, um alerta sonoro é disparado.

🚀 ## Funcionalidades

Detecção facial em tempo real
Cálculo do Eye Aspect Ratio (EAR)
Identificação de olhos fechados
Alerta sonoro automático em caso de sonolência
Visualização dos pontos dos olhos na tela
Exibição do valor do EAR em tempo real

🧠 ## Como funciona

O sistema utiliza landmarks faciais para identificar pontos específicos ao redor dos olhos.

O cálculo do EAR é feito com base na fórmula:

EAR = (||P2 - P6|| + ||P3 - P5||) / (2 * ||P1 - P4||)
Valores baixos de EAR indicam olhos fechados
Se o EAR ficar abaixo de um limite por um tempo contínuo → sonolência detectada

🛠️ ## Tecnologias utilizadas
Python
OpenCV
MediaPipe
Chime (para alertas sonoros)

📄 ## Licença

Este projeto é open-source e pode ser utilizado livremente para fins educacionais e experimentais.
