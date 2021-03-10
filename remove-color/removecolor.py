from PIL import Image

img = Image.open('normal-colorida.jpg').convert('LA')
img.save('normal-preto-e-branco.png')
