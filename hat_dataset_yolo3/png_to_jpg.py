import os
import cv2
from absl import flags, app

# path = os.path.abspath(__file__)
# cwd = os.path.split(path)[0]
# if cwd.endswith('apps'):
#     os.chdir(cwd[0:-4])
#     cwd = os.getcwd()

flags.DEFINE_string('image_path', None, "image to be transformed")
flags.DEFINE_string('output_path', None, 'image save path')
Flags = flags.FLAGS


def png_to_jpg(argv=()):
    del argv
    imagepath = Flags.image_path
    outpath = Flags.output_path
    imagepath = os.path.abspath(imagepath)
    outpath = os.path.abspath(outpath)
    image_name_list = os.listdir(imagepath)
    for i, item in enumerate(image_name_list):
        image_path =imagepath + "\\" + item
        if image_path.endswith(".png"):
            image = cv2.imread(image_path)
            new_image_name = image_path.replace(".png", '.jpg')
            cv2.imwrite(new_image_name, image)
    print("transform completed")


if __name__ == '__main__':
    flags.mark_flag_as_required('image_path')
    flags.mark_flag_as_required('output_path')

    # path = "./raw_image"
    # output_path = "./raw_image"
    app.run(png_to_jpg)
