from gosth_free import execute
from scipy import ndimage, misc


if __name__ == "__main__":
    fmt = 'DeepHDR'     # nombre de la carpeta que esta dentro de 'image_set' que tienen las images
    res = execute(fmt)
    misc.imsave("res/DeepHDR_sin.jpg", res)
    print("Finished...!")
