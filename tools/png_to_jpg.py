import os
import cv2
from absl import flags, app

# path = os.path.abspath(__file__)
# cwd = os.path.split(path)[0]
# if cwd.endswith('apps'):
#     os.chdir(cwd[0:-4])
#     cwd = os.getcwd()

flags.DEFINE_string('image_path', 'D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\data(single_no_hat)\\train_images', "image to be transformed")
flags.DEFINE_string('output_path', 'D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\data(single_no_hat)\\train_images', 'image save path')
flags.DEFINE_bool("overrite", True, "whether to overrite png images with jpg image")
Flags = flags.FLAGS


def png_to_jpg(argv=()):
    del argv
    images_path = Flags.image_path
    overrite = Flags.overrite
    output_path = images_path if overrite else Flags.output_path

    image_name_list = os.listdir(images_path)
    png_nums = 0
    for i, item in enumerate(image_name_list):
        image_path = os.path.join(images_path, item)
        if image_path.endswith(".png"):
            image = cv2.imread(image_path)
            image_name = os.path.split(image_path)[1]
            new_image_name = image_name.replace("png", "jpg")
            new_image_path = os.path.join(output_path, new_image_name)
            cv2.imwrite(new_image_path, image)
            png_nums += 1
            if overrite:
                os.remove(image_path)
                print("overrite %s" % image_name)
            else:
                print("processing on : ", image_name)
    print("transform %d images completed" % png_nums)


if __name__ == '__main__':
    flags.mark_flag_as_required('image_path')
    flags.mark_flag_as_required('output_path')
    flags.mark_flag_as_required('overrite')

    # path = "./raw_image"
    # output_path = "./raw_image"
    app.run(png_to_jpg)
