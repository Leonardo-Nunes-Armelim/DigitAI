import os
import numpy as np
import pygame as pg
from PIL import Image
import tensorflow as tf

loaded_model = tf.keras.models.load_model('mnist_model.keras')

paleta = [(70, 0, 70, 255), (70, 1, 71, 255), (69, 2, 71, 255), (69, 3, 72, 255), (69, 4, 72, 255), (68, 6, 73, 255), (68, 7, 73, 255), (68, 8, 74, 255), (67, 9, 74, 255), (67, 10, 75, 255), (67, 11, 76, 255), (67, 12, 76, 255), (66, 13, 77, 255), (66, 14, 77, 255), (66, 15, 78, 255), (65, 17, 78, 255), (65, 18, 79, 255), (65, 19, 79, 255), (64, 20, 80, 255), (64, 21, 80, 255), (64, 22, 81, 255), (63, 23, 82, 255), (63, 24, 82, 255), (63, 25, 83, 255), (62, 26, 83, 255), (62, 28, 84, 255), (62, 29, 84, 255), (61, 30, 85, 255), (61, 31, 85, 255), (61, 32, 86, 255), (61, 33, 87, 255), (60, 34, 87, 255), (60, 35, 88, 255), (60, 36, 88, 255), (59, 37, 89, 255), (59, 39, 89, 255), (59, 40, 90, 255), (58, 41, 90, 255), (58, 42, 91, 255), (58, 43, 91, 255), (57, 44, 92, 255), (57, 45, 93, 255), (57, 46, 93, 255), (56, 47, 94, 255), (56, 49, 94, 255), (56, 50, 95, 255), (56, 51, 95, 255), (55, 52, 96, 255), (55, 53, 96, 255), (55, 54, 97, 255), (54, 55, 98, 255), (54, 56, 98, 255), (54, 57, 99, 255), (53, 58, 99, 255), (53, 60, 100, 255), (53, 61, 100, 255), (52, 62, 101, 255), (52, 63, 101, 255), (52, 64, 102, 255), (51, 65, 103, 255), (51, 66, 103, 255), (51, 67, 104, 255), (50, 68, 104, 255), (50, 69, 105, 255), (50, 71, 105, 255), (50, 72, 106, 255), (49, 73, 106, 255), (49, 74, 107, 255), (49, 75, 107, 255), (48, 76, 108, 255), (48, 77, 109, 255), (48, 78, 109, 255), (47, 79, 110, 255), (47, 80, 110, 255), (47, 82, 111, 255), (46, 83, 111, 255), (46, 84, 112, 255), (46, 85, 112, 255), (45, 86, 113, 255), (45, 87, 114, 255), (45, 88, 114, 255), (44, 89, 115, 255), (44, 90, 115, 255), (44, 91, 116, 255), (44, 93, 116, 255), (43, 94, 117, 255), (43, 95, 117, 255), (43, 96, 118, 255), (42, 97, 119, 255), (42, 98, 119, 255), (42, 99, 120, 255), (41, 100, 120, 255), (41, 101, 121, 255), (41, 103, 121, 255), (40, 104, 122, 255), (40, 105, 122, 255), (40, 106, 123, 255), (39, 107, 123, 255), (39, 108, 124, 255), (39, 109, 125, 255), (39, 110, 125, 255), (38, 111, 126, 255), (38, 112, 126, 255), (38, 114, 127, 255), (37, 115, 127, 255), (37, 116, 128, 255), (37, 117, 128, 255), (36, 118, 129, 255), (36, 119, 130, 255), (36, 120, 130, 255), (35, 121, 131, 255), (35, 122, 131, 255), (35, 123, 132, 255), (34, 125, 132, 255), (34, 126, 133, 255), (34, 127, 133, 255), (33, 128, 134, 255), (33, 129, 134, 255), (33, 130, 135, 255), (33, 131, 136, 255), (32, 132, 136, 255), (32, 133, 137, 255), (32, 134, 137, 255), (31, 136, 138, 255), (31, 137, 138, 255), (31, 138, 139, 255), (30, 139, 139, 255), (30, 140, 140, 255), (32, 141, 139, 255), (33, 141, 138, 255), (35, 142, 137, 255), (37, 143, 137, 255), (39, 144, 136, 255), (40, 144, 135, 255), (42, 145, 134, 255), (44, 146, 133, 255), (45, 146, 132, 255), (47, 147, 131, 255), (49, 148, 131, 255), (51, 148, 130, 255), (52, 149, 129, 255), (54, 150, 128, 255), (56, 151, 127, 255), (58, 151, 126, 255), (59, 152, 125, 255), (61, 153, 125, 255), (63, 153, 124, 255), (64, 154, 123, 255), (66, 155, 122, 255), (68, 155, 121, 255), (70, 156, 120, 255), (71, 157, 119, 255), (73, 158, 119, 255), (75, 158, 118, 255), (76, 159, 117, 255), (78, 160, 116, 255), (80, 160, 115, 255), (82, 161, 114, 255), (83, 162, 113, 255), (85, 162, 112, 255), (87, 163, 112, 255), (88, 164, 111, 255), (90, 165, 110, 255), (92, 165, 109, 255), (94, 166, 108, 255), (95, 167, 107, 255), (97, 167, 106, 255), (99, 168, 106, 255), (100, 169, 105, 255), (102, 170, 104, 255), (104, 170, 103, 255), (106, 171, 102, 255), (107, 172, 101, 255), (109, 172, 100, 255), (111, 173, 100, 255), (112, 174, 99, 255), (114, 174, 98, 255), (116, 175, 97, 255), (118, 176, 96, 255), (119, 177, 95, 255), (121, 177, 94, 255), (123, 178, 94, 255), (125, 179, 93, 255), (126, 179, 92, 255), (128, 180, 91, 255), (130, 181, 90, 255), (131, 181, 89, 255), (133, 182, 88, 255), (135, 183, 88, 255), (137, 184, 87, 255), (138, 184, 86, 255), (140, 185, 85, 255), (142, 186, 84, 255), (143, 186, 83, 255), (145, 187, 82, 255), (147, 188, 82, 255), (149, 189, 81, 255), (150, 189, 80, 255), (152, 190, 79, 255), (154, 191, 78, 255), (155, 191, 77, 255), (157, 192, 76, 255), (159, 193, 76, 255), (161, 193, 75, 255), (162, 194, 74, 255), (164, 195, 73, 255), (166, 196, 72, 255), (168, 196, 71, 255), (169, 197, 70, 255), (171, 198, 70, 255), (173, 198, 69, 255), (174, 199, 68, 255), (176, 200, 67, 255), (178, 200, 66, 255), (180, 201, 65, 255), (181, 202, 64, 255), (183, 203, 64, 255), (185, 203, 63, 255), (186, 204, 62, 255), (188, 205, 61, 255), (190, 205, 60, 255), (192, 206, 59, 255), (193, 207, 58, 255), (195, 208, 58, 255), (197, 208, 57, 255), (198, 209, 56, 255), (200, 210, 55, 255), (202, 210, 54, 255), (204, 211, 53, 255), (205, 212, 52, 255), (207, 212, 51, 255), (209, 213, 51, 255), (210, 214, 50, 255), (212, 215, 49, 255), (214, 215, 48, 255), (216, 216, 47, 255), (217, 217, 46, 255), (219, 217, 45, 255), (221, 218, 45, 255), (222, 219, 44, 255), (224, 219, 43, 255), (226, 220, 42, 255), (228, 221, 41, 255), (229, 222, 40, 255), (231, 222, 39, 255), (233, 223, 39, 255), (235, 224, 38, 255), (236, 224, 37, 255), (238, 225, 36, 255), (240, 226, 35, 255), (241, 226, 34, 255), (243, 227, 33, 255), (245, 228, 33, 255), (247, 229, 32, 255), (248, 229, 31, 255), (250, 230, 30, 255)]

def clear_terminal():
    try:
        os.system('cls')
    except:
        try:
            os.system('clear')
            pass
        except:
            pass

clear_terminal()

screen_mode = True
while screen_mode:
    print('Em qual modo te tela você quer jogar?\n\nTela cheia = 1\nModo janela = 2')
    response = input('\nResposta: ')
    if response != '1' and response != '2':
        clear_terminal()
        print('Me desculpe mas acho que não entendi a sua resposta. Digite 1 ou 2 para escolher o mode de tela.\n')
    else:
        if response == '1':
            window = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        elif response == '2':
            window = pg.display.set_mode((1280, 720), pg.RESIZABLE)
        screen_mode = False

pg.font.init()
fonte = pg.font.SysFont("Courier New", 20, bold=True)
fonte_2 = pg.font.SysFont("Courier New", 50, bold=True)
fonte_3 = pg.font.SysFont("Courier New", 75, bold=True)

# Variáveis de mouse
last_click_status = (False, False, False)

def mouse_has_clicked(input):
    if last_click_status == input:
        return (False, False, False)
    else:
        left_button = False
        center_button = False
        right_button = False
        if last_click_status[0] == False and input[0] == True:
            left_button = True
        if last_click_status[1] == False and input[1] == True:
            center_button = True
        if last_click_status[2] == False and input[2] == True:
            right_button = True

        return (left_button, center_button, right_button)

def clear_window():
    pg.draw.rect(window, (255, 255, 255), (0, 0, window.get_width(), window.get_height()))

def rect_center(outside_rect, inside_rect):
    left_margin = (outside_rect[0] - inside_rect[0]) / 2
    top_margin = (outside_rect[1] - inside_rect[1]) / 2
    return left_margin, top_margin

def get_draw(img, pos, side):
    unit = int(side / 28)

    img_28x28 = np.zeros((28, 28))

    for y in range(28):
        for x in range(28):
            color = window.get_at((int(pos[0]) + (x * unit), int(pos[1]) + (y * unit)))
            img_28x28[y][x] = color[0]

    for y in range(28):
        for x in range(28):
            if y >= 1 and y <= 26 and x >= 1 and x <= 26:
                color = int((img_28x28[y][x] + img_28x28[y-1][x] + img_28x28[y][x+1] + img_28x28[y+1][x] + img_28x28[y][x-1])/5)
                img.putpixel((x, y), (color, color, color, 255))
    img.save('./numb.png')

    img = Image.open('./numb.png')
    x = img.size[0]
    y = img.size[1]
    pixel = img.load()

    # Criando variável com os valores dos pixels em 0
    img_data = np.zeros((y, x))

    # Loop de escrita dos valores dos pixels da imagem
    for yy in range(y):
        for xx in range(x):
            if pixel[xx, yy][0] < 255:
                img_data[yy, xx] = (255 - pixel[xx, yy][0]) / 255

    # Ajuste dos dados da imagem
    data = []
    data.append(img_data)
    data = np.array(data)

    prediction = loaded_model.predict(data)
    print('Predição:', np.argmax(prediction, axis=1))

    # ################################################################################

    # Acessar os pesos de cada camada (verificando se a camada tem pesos e vieses)
    for layer in loaded_model.layers:
        weights = layer.get_weights()
        if weights:
            weights, biases = weights
            if layer.name == 'dense':
                count = 0
                dense = Image.new("RGBA", (337, 337), (255, 255, 255, 255))
                for ww in range(weights.shape[1]):
                    img_28x28 = np.zeros((28, 28))
                    for w in range(weights.shape[0]):
                        img_28x28[w//28][w%28] = (img.getpixel((w%28, w//28))[0] * weights[w][ww]) + biases[ww]
                    # Normalizando valores
                    img_28x28_min = img_28x28.min()
                    img_28x28_max = img_28x28.max()
                    img_28x28_norm = (img_28x28 - img_28x28_min) / (img_28x28_max - img_28x28_min)
                    for y in range(28):
                        for x in range(28):
                            color = paleta[int(img_28x28_norm[y][x] * 255)]
                            dense.putpixel((((count%12)*28) + x, ((count//12)*28) + y), color)
                    for y in range(337):
                        for x in range(337):
                            if x%28 == 0 or y%28 == 0:
                                dense.putpixel((x, y), (255, 0, 0, 255))
                    count += 1
                dense.save('./dense.png')
            if layer.name == 'dense_1':
                count = 0
                dense = Image.new("RGBA", (97, 97), (255, 255, 255, 255))
                for ww in range(weights.shape[1]):
                    img_12x12 = np.zeros((12, 12))
                    for w in range(weights.shape[0]):
                        img_12x12[w//12][w%12] = (img.getpixel((w%12, w//12))[0] * weights[w][ww]) + biases[ww]
                    # Normalizando valores
                    img_12x12_min = img_12x12.min()
                    img_12x12_max = img_12x12.max()
                    img_12x12_norm = (img_12x12 - img_12x12_min) / (img_12x12_max - img_12x12_min)
                    for y in range(12):
                        for x in range(12):
                            color = paleta[int(img_12x12_norm[y][x] * 255)]
                            dense.putpixel((((count%8)*12) + x, ((count//8)*12) + y), color)
                    for y in range(97):
                        for x in range(97):
                            if x%12 == 0 or y%12 == 0:
                                dense.putpixel((x, y), (255, 0, 0, 255))
                    count += 1
                dense.save('./dense_1.png')
    
    return prediction

prediction = np.array([[4.2673491e-09, 4.4654698e-09, 4.3872822e-10, 3.4134337e-04, 1.7633894e-10, 9.9957007e-01, 1.4154314e-07, 1.9176756e-09, 4.4520648e-06, 8.3886145e-05]])
clear = True

img = Image.new("RGBA", (28, 28), (255, 255, 255, 255))

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            quit()
        if event.type == pg.KEYDOWN:
            if pg.key.name(event.key) == 'escape':
                pg.quit()
                quit()

    # Mouse info
    mouse_position  = pg.mouse.get_pos()
    mouse_input = pg.mouse.get_pressed()
    mouse_click = mouse_has_clicked(mouse_input)
    mouse = (mouse_position, mouse_input, mouse_click)

    if clear:
        clear_window()
        clear = False

    height = window.get_height() / 3
    width = window.get_width() / 4
    scale = 0.95

    left_margin_2x1 = (width - (height * scale)) / 2
    top_margin_2x1 = (height - (height * scale)) / 2
    min_side_2x1 = height * scale

    left_margin_2x2 = ((width * 2) - (height * 2 * scale)) / 2
    top_margin_2x2 = ((height * 2) - (height * 2 * scale)) / 2
    min_side_2x2 = height * 2 * scale

    first_layer = (308, 308)
    left_margin, top_margin = rect_center((width, height * 2), first_layer)

    numb_image = pg.image.load('./numb.png')
    numb_image = pg.transform.scale(numb_image, (min_side_2x1, min_side_2x1))
    window.blit(numb_image, (left_margin_2x1, top_margin_2x1))
    pg.draw.rect(window, (0,   0, 0), (left_margin_2x1, top_margin_2x1, min_side_2x1, min_side_2x1), width=3)
    
    # Dense 1
    dense_image = pg.image.load('./dense.png')
    dense_image = pg.transform.scale(dense_image, (min_side_2x1, min_side_2x1))
    window.blit(dense_image, (width + left_margin_2x1, top_margin_2x1))
    pg.draw.rect(window, (255, 0, 0), (width     + left_margin_2x1, top_margin_2x1, min_side_2x1, min_side_2x1), width=1)

    # Dense 2
    dense_1_image = pg.image.load('./dense_1.png')
    dense_1_image = pg.transform.scale(dense_1_image, (min_side_2x1, min_side_2x1))
    window.blit(dense_1_image, (width * 2 + left_margin_2x1, top_margin_2x1))
    pg.draw.rect(window, (255, 0, 0), (width * 2 + left_margin_2x1, top_margin_2x1, min_side_2x1, min_side_2x1), width=1)

    # Predicted Number
    x = width * 3 + left_margin_2x1
    y = top_margin_2x1
    side = min_side_2x1 / 5
    # Pinta de branco atrás da porcentagens da previsão
    pg.draw.rect(window, (255, 255, 255), (width * 3, 0, height * 2, height))
    # 0
    pg.draw.circle(window, paleta[int(prediction[0][0] * 255)], (x - 15, y + (side * 0.5)), 10)
    pg.draw.circle(window, (0, 0, 0), (x - 15, y + (side * 0.5)), 10, width=2)
    texto = fonte.render(f'0: {round(prediction[0][0] * 100 , 1)}%', 1, (0, 0, 0))
    window.blit(texto, (x, y + (side * 0.5) - (texto.get_size()[1] / 2)))
    # 1
    pg.draw.circle(window, paleta[int(prediction[0][1] * 255)], (x - 15, y + (side * 1.5)), 10)
    pg.draw.circle(window, (0, 0, 0), (x - 15, y + (side * 1.5)), 10, width=2)
    texto = fonte.render(f'1: {round(prediction[0][1] * 100, 1)}%', 1, (0, 0, 0))
    window.blit(texto, (x, y + (side * 1.5) - (texto.get_size()[1] / 2)))
    # 2
    pg.draw.circle(window, paleta[int(prediction[0][2] * 255)], (x - 15, y + (side * 2.5)), 10)
    pg.draw.circle(window, (0, 0, 0), (x - 15, y + (side * 2.5)), 10, width=2)
    texto = fonte.render(f'2: {round(prediction[0][2] * 100, 1)}%', 1, (0, 0, 0))
    window.blit(texto, (x, y + (side * 2.5) - (texto.get_size()[1] / 2)))
    # 3
    pg.draw.circle(window, paleta[int(prediction[0][3] * 255)], (x - 15, y + (side * 3.5)), 10)
    pg.draw.circle(window, (0, 0, 0), (x - 15, y + (side * 3.5)), 10, width=2)
    texto = fonte.render(f'3: {round(prediction[0][3] * 100, 1)}%', 1, (0, 0, 0))
    window.blit(texto, (x, y + (side * 3.5) - (texto.get_size()[1] / 2)))
    # 4
    pg.draw.circle(window, paleta[int(prediction[0][4] * 255)], (x - 15, y + (side * 4.5)), 10)
    pg.draw.circle(window, (0, 0, 0), (x - 15, y + (side * 4.5)), 10, width=2)
    texto = fonte.render(f'4: {round(prediction[0][4] * 100, 1)}%', 1, (0, 0, 0))
    window.blit(texto, (x, y + (side * 4.5) - (texto.get_size()[1] / 2)))
    # 5
    pg.draw.circle(window, paleta[int(prediction[0][5] * 255)], (x + 10 + (side * 5 / 2), y + (side * 0.5)), 10)
    pg.draw.circle(window, (0, 0, 0), (x + 10 + (side * 5 / 2), y + (side * 0.5)), 10, width=2)
    texto = fonte.render(f'5: {round(prediction[0][5] * 100, 1)}%', 1, (0, 0, 0))
    window.blit(texto, (x + 25 + (side * 5 / 2), y + (side * 0.5) - (texto.get_size()[1] / 2)))
    # 6
    pg.draw.circle(window, paleta[int(prediction[0][6] * 255)], (x + 10 + (side * 5 / 2), y + (side * 1.5)), 10)
    pg.draw.circle(window, (0, 0, 0), (x + 10 + (side * 5 / 2), y + (side * 1.5)), 10, width=2)
    texto = fonte.render(f'6: {round(prediction[0][6] * 100, 1)}%', 1, (0, 0, 0))
    window.blit(texto, (x + 25 + (side * 5 / 2), y + (side * 1.5) - (texto.get_size()[1] / 2)))
    # 7
    pg.draw.circle(window, paleta[int(prediction[0][7] * 255)], (x + 10 + (side * 5 / 2), y + (side * 2.5)), 10)
    pg.draw.circle(window, (0, 0, 0), (x + 10 + (side * 5 / 2), y + (side * 2.5)), 10, width=2)
    texto = fonte.render(f'7: {round(prediction[0][7] * 100, 1)}%', 1, (0, 0, 0))
    window.blit(texto, (x + 25 + (side * 5 / 2), y + (side * 2.5) - (texto.get_size()[1] / 2)))
    # 8
    pg.draw.circle(window, paleta[int(prediction[0][8] * 255)], (x + 10 + (side * 5 / 2), y + (side * 3.5)), 10)
    pg.draw.circle(window, (0, 0, 0), (x + 10 + (side * 5 / 2), y + (side * 3.5)), 10, width=2)
    texto = fonte.render(f'8: {round(prediction[0][8] * 100, 1)}%', 1, (0, 0, 0))
    window.blit(texto, (x + 25 + (side * 5 / 2), y + (side * 3.5) - (texto.get_size()[1] / 2)))
    # 9
    pg.draw.circle(window, paleta[int(prediction[0][9] * 255)], (x + 10 + (side * 5 / 2), y + (side * 4.5)), 10)
    pg.draw.circle(window, (0, 0, 0), (x + 10 + (side * 5 / 2), y + (side * 4.5)), 10, width=2)
    texto = fonte.render(f'9: {round(prediction[0][9] * 100, 1)}%', 1, (0, 0, 0))
    window.blit(texto, (x + 25 + (side * 5 / 2), y + (side * 4.5) - (texto.get_size()[1] / 2)))

    # Predicted Box
    pg.draw.rect(window, (255, 255, 255), ((width * 2) + (((width * 2) - (width * 2 * scale)) / 2), height + (((height * 2) - (height * 2 * scale)) / 2),  width * 2 * scale, height * 2 * scale))

    # Predict Button
    predict_btn_x = (width * 2) + (((width * 2) - (width * 2 * scale)) / 2)
    predict_btn_y = height + (((height * 2) - (height * 2 * scale)) / 2)
    predict_btn_w = 250
    predict_btn_h = 75
    if mouse[0][0] >= predict_btn_x and mouse[0][0] <= predict_btn_x + predict_btn_w and mouse[0][1] >= predict_btn_y and mouse[0][1] <= predict_btn_y + predict_btn_h:
        pg.draw.rect(window, (0, 255, 0), (predict_btn_x, predict_btn_y,  predict_btn_w, predict_btn_h))
        if mouse[2][0]:
            prediction = get_draw(img, position, height * 2 * scale)
    else:
        pg.draw.rect(window, (0, 200, 0), (predict_btn_x, predict_btn_y,  predict_btn_w, predict_btn_h))
    pg.draw.rect(window, (0, 0, 0), (predict_btn_x, predict_btn_y,  predict_btn_w, predict_btn_h), width=5)
    texto = fonte_2.render('Predict', 1, (0, 0, 0))
    window.blit(texto, (predict_btn_x + (predict_btn_w - texto.get_size()[0]) / 2, predict_btn_y + (predict_btn_h - texto.get_size()[1]) / 2))
    
    # Clear Button
    clear_btn_x = (width * 3) + (((width * 3) - (width * 3 * scale)) / 2) + 25
    clear_btn_y = height + (((height * 2) - (height * 2 * scale)) / 2)
    clear_btn_w = 250
    clear_btn_h = 75
    if mouse[0][0] >= clear_btn_x and mouse[0][0] <= clear_btn_x + clear_btn_w and mouse[0][1] >= clear_btn_y and mouse[0][1] <= clear_btn_y + clear_btn_h:
        pg.draw.rect(window, (255, 0, 0), (clear_btn_x, clear_btn_y,  clear_btn_w, clear_btn_h))
        if mouse[2][0]:
            clear = True
    else:
        pg.draw.rect(window, (200, 0, 0), (clear_btn_x, clear_btn_y,  clear_btn_w, clear_btn_h))
    pg.draw.rect(window, (0, 0, 0), (clear_btn_x, clear_btn_y,  clear_btn_w, clear_btn_h), width=5)
    texto = fonte_2.render('Clear', 1, (0, 0, 0))
    window.blit(texto, (clear_btn_x + (clear_btn_w - texto.get_size()[0]) / 2, clear_btn_y + (clear_btn_h - texto.get_size()[1]) / 2))

    # Resposta da previsão
    prediction_x = width * 2
    prediction_y = height + (((height * 2) - (height * 2 * scale)) / 2)
    prediction_w = width * 2
    prediction_h = height * 2
    texto_1 = fonte_3.render('Predicted', 1, (0, 0, 0))
    window.blit(texto_1, (prediction_x + (prediction_w - texto_1.get_size()[0]) / 2, prediction_y + (prediction_h - texto_1.get_size()[1]) / 2 - 70))
    texto_2 = fonte_3.render('number', 1, (0, 0, 0))
    window.blit(texto_2, (prediction_x + (prediction_w - texto_2.get_size()[0]) / 2, prediction_y + (prediction_h - texto_2.get_size()[1]) / 2))
    texto_3 = fonte_3.render(f'{np.argmax(prediction, axis=1)[0]}', 1, (0, 0, 0))
    window.blit(texto_3, (prediction_x + (prediction_w - texto_3.get_size()[0]) / 2, prediction_y + (prediction_h - texto_3.get_size()[1]) / 2 + 70))

    # Draw Box
    pg.draw.rect(window, (0, 0, 0), (left_margin_2x2 - 3, height + top_margin_2x2 - 3, height * 2 * scale + 3, height * 2 * scale + 3), width=3)

    # Box de desenho do Número
    if mouse[0][0] > left_margin_2x2 and mouse[0][0] < left_margin_2x2 + (height * 2 * scale):
        if mouse[0][1] > height + top_margin_2x2 and mouse[0][1] < height + top_margin_2x2 + (height * 2 * scale):
            if mouse[1][0]:
                pg.draw.line(window, (0, 0, 0), last_mouse_pos, mouse[0], width=30)

    # Variaveis pivot

    position = (left_margin_2x2, height + top_margin_2x2)

    x = (left_margin_2x2 + (height * 2 * scale)) / 2
    y = (height + top_margin_2x2 + (height * 2 * scale)) / 2

    last_mouse_pos = mouse[0]
    last_click_status = mouse_input

    pg.display.update()