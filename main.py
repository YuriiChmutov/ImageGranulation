import cv2
import numpy as np
from scipy.spatial import distance
from scipy.linalg import hadamard

from descriptor import Descriptor


# to convert number to bit array 8 -> [0, 0, 0, 0, 1, 0, 0, 0]
def bitfield(n):
    result = [int(digit) for digit in bin(n)[2:]]
    len_result = len(result)
    new_result = []
    if len_result < 8:
        amount_of_zeros = 8 - len_result
        for i in range(amount_of_zeros):
            new_result.append(0)
        for i in range(len(result)):
            new_result.append(result[i])
        return new_result
    return result


# to convert 32-byte format descriptor to 256-bit format
def return_array_of_256_bits(point):
    result_array = []
    for byte in range(len(point)):
        bit_point = bitfield(point[byte])
        for x in range(len(bit_point)):
            result_array.append(bit_point[x])
    return result_array


# to convert all 32-byte format descriptors to 256-bit format
def convert_32_descriptors_to_256_bit(descriptors):
    result_matrix = []
    for keypoint in range(len(descriptors)):
        a = return_array_of_256_bits(descriptors[keypoint])
        result_matrix.append(a)
    return result_matrix


def process_image(image_path, class_name, descriptors_amount=20):
    img = cv2.imread(image_path)
    orb = cv2.ORB_create(nfeatures=descriptors_amount)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    descriptors_bit_format = convert_32_descriptors_to_256_bit(descriptors)
    descriptors_list = [Descriptor(descriptor, False, index, class_name) for index, descriptor in enumerate(descriptors_bit_format)]
    return descriptors_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    image_data = [
        ('images/Liverpool.jpg', 'A'),
        ('images/Leicester.jpg', 'B'),
        ('images/BayernMunchen.jpg', 'C'),
        ('images/Eintracht.jpg', 'D'),
        ('images/Brentford.jpg', 'E')
    ]

    # img_A = cv2.imread('images/Liverpool.jpg')
    # img_B = cv2.imread('images/Leicester.jpg')
    # img_C = cv2.imread('images/BayernMunchen.jpg')
    # img_D = cv2.imread('images/Eintracht.jpg')
    # img_E = cv2.imread('images/Brentford.jpg')
    # img_F = cv2.imread('images/ManchesterCity.jpg')

    # cv2.imshow("Lion_simple", img)
    # cv2.waitKey(0)

    # descriptors_amount = 500
    # orb = cv2.ORB_create(nfeatures=descriptors_amount)

    # keypoints_img_A, descriptors_img_A = orb.detectAndCompute(img_A, None)
    # keypoints_img_B, descriptors_img_B = orb.detectAndCompute(img_B, None)
    # keypoints_img_C, descriptors_img_C = orb.detectAndCompute(img_C, None)
    # keypoints_img_D, descriptors_img_D = orb.detectAndCompute(img_D, None)
    # keypoints_img_E, descriptors_img_E = orb.detectAndCompute(img_E, None)

    # img_s = cv2.drawKeypoints(img, keypoints_img, None)
    # cv2.imshow("Lion noisy with key points", img_s)
    # cv2.waitKey(0)

    # descriptors_img_bit_format_A = convert_32_descriptors_to_256_bit(descriptors_img_A)
    # descriptors_img_bit_format_B = convert_32_descriptors_to_256_bit(descriptors_img_B)
    # descriptors_img_bit_format_C = convert_32_descriptors_to_256_bit(descriptors_img_C)
    # descriptors_img_bit_format_D = convert_32_descriptors_to_256_bit(descriptors_img_D)
    # descriptors_img_bit_format_E = convert_32_descriptors_to_256_bit(descriptors_img_E)
    #
    # descriptors_list_A = \
    #     [Descriptor(descriptor, False, index, "A") for index, descriptor in enumerate(descriptors_img_bit_format_A)]
    # descriptors_list_B = \
    #     [Descriptor(descriptor, False, index, "B") for index, descriptor in enumerate(descriptors_img_bit_format_B)]
    # descriptors_list_C = \
    #     [Descriptor(descriptor, False, index, "C") for index, descriptor in enumerate(descriptors_img_bit_format_C)]
    # descriptors_list_D = \
    #     [Descriptor(descriptor, False, index, "D") for index, descriptor in enumerate(descriptors_img_bit_format_D)]
    # descriptors_list_E = \
    #     [Descriptor(descriptor, False, index, "E") for index, descriptor in enumerate(descriptors_img_bit_format_E)]
    #
    # descriptors_combined_etalons = descriptors_list_A + descriptors_list_B + descriptors_list_C + descriptors_list_D + descriptors_list_E

    descriptors_combined_etalons = []
    for image_path, class_name in image_data:
        descriptors_combined_etalons.extend(process_image(image_path, class_name))

    print(f'descriptors_combined_etalons len: {len(descriptors_combined_etalons)}')

    class_counts = {class_name: 0 for _, class_name in image_data}

    # Loop through each image
    for image_path, class_name in image_data:
        # Process the image and get descriptors
        descriptors_list = process_image(image_path, class_name)

        # Iterate over descriptors of the image
        for descriptor in descriptors_list:
            find = descriptor.find_class_of_closest_descriptor_by_hamming_distance(descriptors_combined_etalons)
            if find in class_counts:
                class_counts[find] += 1

        # for class_label, count in class_counts.items():
        #     print(f'{class_label}: {count} / {len(descriptors_list)}')

    # Print the counts for each class
    for class_label, count in class_counts.items():
        print(f'{class_label}: {count}')

    # A = 0
    # B = 0
    # C = 0
    # D = 0
    # E = 0
    #
    # for i in range(len(descriptors_list_A)):
    #
    #     find = descriptors_list_A[i].find_class_of_closest_descriptor_by_hamming_distance(descriptors_combined_etalons)
    #
    #     if find == "A":
    #         A = A + 1
    #     elif find == "B":
    #         B = B + 1
    #     elif find == "C":
    #         C = C + 1
    #     elif find == "D":
    #         D = D + 1
    #     elif find == "E":
    #         E = E + 1
    #
    # print(f'A: {A}; B: {B}; C: {C}; D: {D}; E: {E};')

    # find = descriptors_list_A[0].find_class_of_closest_descriptor_by_hamming_distance(descriptors_combined_etalons)

    # print(find)
    #------------------------------I level---------------------------------------#


    # for descriptor in descriptors_list_A:
    #     descriptor.mark_closest_descriptors(descriptors_list_A)
    #
    # marked_count_A = sum(1 for descriptor in descriptors_list_A if descriptor.marked)
    # print("I рівень, кількість відмічених дескрипторів:", marked_count_A)
    # print("I рівень, кількість дескрипторів що залишилась:", len(descriptors_list_A) - marked_count_A)

    #------------------------------II level---------------------------------------#

    # print('-----------------------------------------------------------------------------')
    #
    # unmarked_descriptors_A = [descriptor for descriptor in descriptors_list_A if not descriptor.marked]
    #
    # for descriptor in unmarked_descriptors_A:
    #     descriptor.mark_closest_descriptors(unmarked_descriptors_A)
    #
    # marked_count = sum(1 for descriptor in unmarked_descriptors_A if descriptor.marked)
    # print("II рівень, кількість відмічених дескрипторів:", marked_count)
    # print("II рівень, кількість дескрипторів що залишилась:", len(unmarked_descriptors_A) - marked_count_A)












