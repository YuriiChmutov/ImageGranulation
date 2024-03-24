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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread('images/Liverpool.jpg')

    cv2.imshow("Lion_simple", img)
    # cv2.waitKey(0)

    descriptors_amount = 500
    orb = cv2.ORB_create(nfeatures=descriptors_amount)

    keypoints_img, descriptors_img = orb.detectAndCompute(img, None)

    img_s = cv2.drawKeypoints(img, keypoints_img, None)
    cv2.imshow("Lion noisy with key points", img_s)
    # cv2.waitKey(0)

    print(f'Image contains {descriptors_amount} descriptors\n')
    print(f'Default length of ORB descriptor = {len(descriptors_img[0])}')

    descriptors_img_bit_format = convert_32_descriptors_to_256_bit(descriptors_img)
    print(f'\nLength of converted to bit format ORB descriptor = {len(descriptors_img_bit_format[0])}')
    print(f'Converted to bit format ORB descriptor looks like: {descriptors_img_bit_format[0]}')

    descriptors_list = [Descriptor(descriptor, False, index) for index, descriptor in enumerate(descriptors_img_bit_format)]
    # for descriptor in descriptors_list[:5]:
    #     print("Descriptor:", descriptor.descriptor)
    #     print("Marked:", descriptor.marked)
    #     print("Index:", descriptor.index)

    for descriptor in descriptors_list:
        descriptor.mark_closest_descriptors(descriptors_list)

    marked_count = sum(1 for descriptor in descriptors_list if descriptor.marked)
    print("Number of marked descriptors:", marked_count)

    marked_indexes = [descriptor.index for descriptor in descriptors_list if descriptor.marked]
    print("Collection of marked indexes:", marked_indexes)

    ################################

    print('-------------------------------------------')

    unmarked_descriptors = [descriptor for descriptor in descriptors_list if not descriptor.marked]

    print(f'Unmarked descriptors len: {len(unmarked_descriptors)}')

    for descriptor in unmarked_descriptors:
        descriptor.mark_closest_descriptors(unmarked_descriptors)

    marked_count = sum(1 for descriptor in unmarked_descriptors if descriptor.marked)
    print("Number of marked descriptors:", marked_count)

    marked_indexes = [descriptor.index for descriptor in unmarked_descriptors if descriptor.marked]
    print("Collection of marked indexes:", marked_indexes)

    # print('-------------')

    # diff_count = sum(1 for i in range(len(descriptors_list[1].descriptor)) if descriptors_list[4].descriptor[i] != descriptors_list[14].descriptor[i])
    # print("Количество отличающихся элементов:", diff_count)











