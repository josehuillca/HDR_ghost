import argparse
from gosth_free import execute
from imageio import imwrite


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fmt', type=str, default='tren2')
    args = parser.parse_args()

    fmt = args.fmt     # name of the folder that is inside 'image_set', which contains the images to be processed.
    res = execute(fmt)
    save_path = "res/" + fmt + "_result.jpg"
    print("SAVE IMAGE: ", save_path)
    imwrite(save_path, res)
    print("Finished...!")
