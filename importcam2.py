
import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)
reconhecimento_rosto = mp.solutions.face_detection
desenho = mp.solutions.drawing_utils

reconhecedor_rosto = reconhecimento_rosto.FaceDetection(min_detection_confidence=0.5)

while webcam.isOpened():
    validacao, frame = webcam.read()
    if not validacao:
        break

    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultado = reconhecedor_rosto.process(imagem_rgb)

    if resultado.detections:
        for detecao in resultado.detections:
          
          caixa_delimitadora = detecao.location_data.relative_bounding_box
          altura, largura, _ = frame.shape
          pontos = [
                int(caixa_delimitadora.xmin * largura),
                int(caixa_delimitadora.ymin * altura),
                int(caixa_delimitadora.width * largura),
                int(caixa_delimitadora.height * altura),
            ]

          desenho.draw_detection(frame, detecao)
          cv2.putText(frame, f"{int(detecao.score[0] * 100)}%", (pontos[0], pontos [1] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2 )
          
        cv2.imshow("Rostos na sua webcam", frame)
        if cv2.waitKey(5) == 27:
             break

webcam.release()
cv2.destroyAllWindows()