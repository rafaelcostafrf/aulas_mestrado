import cv2 as cv

cap = (cv.VideoCapture(0), cv.VideoCapture(1))
names = ('c920', 'c270')
print('Pressione \'s\' para salvar \'q\' para sair da câmera')
for cam, name  in zip(cap, names):
    saved = 0
    caminho_salva = './imagens/'+name+'/'
    while saved < 15:
        ret, frame = cam.read()       
        if ret:
            cv.imshow('captura', frame)
            key = cv.waitKey(1)
            if key == ord('s'):
                nome = caminho_salva+str(saved)+'.jpg'
                cv.imwrite(nome, frame)
                print(f'Imagem n.{saved} salva em {nome}')
                saved += 1
            if key == ord('q'):
                break
    print('Câmera Finalizada')
    cam.release()
cv.destroyAllWindows()