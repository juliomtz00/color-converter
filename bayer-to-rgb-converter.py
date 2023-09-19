import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_image", type=str, default='calibration-images', help="Folder where the calibration images are")
parser.add_argument('--conversion', required=True, type=str.lower, choices = ["bayer2rgb","rgb2bayer"])
parser.add_argument('--option', required=False, type=str.lower, default = "bggr", choices = ['bggr', 'gbrg','grgb','rggb'])
args = parser.parse_args()

def pixel (img):
    img = img.astype(np.float64) 
    pixel = lambda x,y : {
        0: [ img[x][y] , (img[x][y-1] + img[x-1][y] + img[x+1][y] + img[x][y+1]) / 4 ,  (img[x-1][y-1] + img[x+1][y-1] + img[x-1][y+1] + img[x+1][y+1]) / 4 ] ,
        1: [ (img[x-1][y] + img[x+1][y])  / 2,img[x][y] , (img[x][y-1] + img[x][y+1]) / 2 ],
        2: [(img[x][y-1] + img[x][y+1]) / 2 ,img[x][y], (img[x-1][y] + img[x+1][y]) / 2],
        3: [(img[x-1][y-1] + img[x+1][y-1] + img[x-1][y+1] + img[x+1][y+1]) / 4 , (img[x][y-1] + img[x-1][y] + img[x+1][y] + img[x][y+1]) / 4 ,img[x][y] ]
    } [  x % 2 + (y % 2)*2]
    res = np.zeros ( [    np.size(img,0) , np.size(img,1)  , 3] )
    for x in range (1,np.size(img,0)-2):
        for y in range (1,np.size(img,1)-2):
            p = pixel(x,y)
            p.reverse()
            res[x][y] = p
    res = res.astype(np.uint8)
    return res

def channel_break (img):
    img = img.astype(np.float64) 
    red=np.copy (img);red [1::2,:]=0;red[:,1::2]=0
    blue=np.copy (img);blue [0::2,:]=0;blue[:,0::2]=0
    green=np.copy (img);green [0::2,0::2]=0;green [1::2,1::2]=0
    red = red.astype(np.float64) 
    blue = blue.astype(np.float64) 
    green = green.astype(np.float64) 
    return (red,green,blue)

def show_image(label,img):
    cv2.namedWindow(label, cv2.WINDOW_NORMAL)
    cv2.imshow(label,img)

def covert_to_bayer(image,opc):

    rgb_image = cv2.imread(image, cv2.IMREAD_COLOR)
    #gray_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
    gray_image = rgb_image
    #height, width = gray_image.shape
    height=gray_image.shape[0]
    width=gray_image.shape[1]
    # Crear imagen en modo '1' (blanco y negro)

    bayer_image= np.zeros((height,width,3), dtype = "uint8")


    y1=range(0,height,2)
    x1=range(0,width,2)

    # Convertir a Bayer
    if opc=='grgb':
        for y in y1[0:(len(y1))]:
            for x in x1[0:(len(x1))]:

                # Obtener valores de los cuatro píxeles
                red = gray_image[y, x + 1,2]
                green1 = gray_image[y, x,1]
                #print(x,y)
                blue = gray_image[y + 1, x,0]
                green2 = gray_image[y + 1, x + 1,0]

                # Promediar los valores de los píxeles verdes
                #green = (green1 + green2) / 2
                # Asignar valores de píxeles a la imagen Bayer
                bayer_image[y, x,1] = green1
                bayer_image[y, (x + 1),2] = red
                bayer_image[(y + 1), x,0] = blue
                bayer_image[(y + 1), (x + 1),1] = green2



    elif opc=='bggr':

        for y in y1[0:(len(y1))]:
            for x in x1[0:(len(x1))]:

                # Obtener valores de los cuatro píxeles
                green1 = gray_image[y, x + 1,1]
                blue = gray_image[y, x,0]
                green2 = gray_image[y + 1, x,1]
                red = gray_image[y + 1, x + 1,2]

                # Promediar los valores de los píxeles verdes

                # Asignar valores de píxeles a la imagen Bayer
                bayer_image[y, x,0] = blue
                bayer_image[y, (x + 1),1] = green1
                bayer_image[(y + 1), x,1] = green2
                bayer_image[(y + 1), (x + 1),2] = red
    elif opc=='gbrg':

        for y in y1[0:(len(y1))]:
            for x in x1[0:(len(x1))]:

                # Obtener valores de los cuatro píxeles
                blue = gray_image[y, x + 1,0]
                green1 = gray_image[y, x,1]
                red= gray_image[y + 1, x,2]
                green2 = gray_image[y + 1, x + 1,1]

                # Promediar los valores de los píxeles verdes

                # Asignar valores de píxeles a la imagen Bayer
                bayer_image[y, x,1] = green1
                bayer_image[y, (x + 1),0] = blue
                bayer_image[(y + 1), x,2] = red
                bayer_image[(y + 1), (x + 1),1] = green2
    elif opc=='rggb':

        for y in y1[0:(len(y1))]:
            for x in x1[0:(len(x1))]:

                # Obtener valores de los cuatro píxeles
                green1 = gray_image[y, x + 1,1]
                red = gray_image[y, x,2]
                green2= gray_image[y + 1, x,1]
                blue = gray_image[y + 1, x + 1,0]

                # Promediar los valores de los píxeles verdes

                # Asignar valores de píxeles a la imagen Bayer
                bayer_image[y, x,2] = red
                bayer_image[y, (x + 1),1] = green1
                bayer_image[(y + 1), x,1] = green2
                bayer_image[(y + 1), (x + 1),0] = blue

    return bayer_image

if args.conversion == "bayer2rgb":

    imrows = 540
    imcols = 600
    imsize = imrows*imcols
    with open(args.input_image, "rb") as rawimage:
        bayer_img = np.fromfile(rawimage, np.dtype('u1'), imsize).reshape((imrows, imcols)) 

    # plot bayer imager
    show_image("Bayer Image",bayer_img)

    # this algorithm conversion
    rgb_res = pixel (bayer_img)
    show_image('the article conversion',rgb_res)

    # open cv conversion
    colour = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_BG2BGR)
    show_image('color image by open cv',colour)

    # break to RGB  channels
    RGB = channel_break(bayer_img)
    blue_only = pixel (RGB[2])
    show_image('blue only',blue_only)

    green_only = pixel (RGB[1])
    show_image('green only',green_only)

    red_only = pixel (RGB[0])
    show_image('red only',red_only)

else:
    # plot bayer imager
    rgb_image = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
    show_image("RGB Image",rgb_image)

    img = args.input_image
    b = covert_to_bayer(img,args.option)

    show_image("Bayer Image",b)


# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
