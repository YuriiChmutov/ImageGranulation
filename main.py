import cv2
import numpy as np
import time
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


def compare_descriptors(descriptors, combined_descriptors, class_counts):
    # print(f'Comparing descriptors for image {image_path} with combined set...')
    num_combined_descriptors = len(combined_descriptors)

    for descriptor in descriptors:
        closest_class = descriptor.find_class_of_closest_descriptor_by_hamming_distance(combined_descriptors)
        class_counts[closest_class] += 1

        # print(f'  Descriptor {descriptor.index} compared with {num_combined_descriptors} combined descriptors.')
    
    print()
    print('Class counts:')
    print(class_counts)
    print()


def print_class_counts(image_path, class_counts):
    print(f'Image {image_path} class counts:')
    for class_label, count in class_counts.items():
        print(f'{class_label}: {count}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    image_input = [
        ('images/Lion.jpg', 'input_A'),
        ('images/ManchesterCity.jpg', 'input_B'),
        ('images/Rangers.jpg', 'input_C')
        # ('images/Liverpool.jpg', 'A'),
        # ('images/Leicester.jpg', 'B'),
        # ('images/BayernMunchen.jpg', 'C'),
        # ('images/Eintracht.jpg', 'D'),
        # ('images/Brentford.jpg', 'E')
    ]

    image_etalons = [ #image_data
        ('images/Liverpool.jpg', 'A'),
        ('images/Leicester.jpg', 'B'),
        ('images/BayernMunchen.jpg', 'C'),
        ('images/Eintracht.jpg', 'D'),
        ('images/Brentford.jpg', 'E')
    ]

    descriptors_by_image_input = {}
    descriptors_by_image_etalons = {}

    descriptors_amount = 500

    for image_path, class_name in image_input:
        descriptors_by_image_input[image_path] = process_image(image_path, class_name, descriptors_amount)

    # Process images in image_etalons
    for image_path, class_name in image_etalons:
        descriptors_by_image_etalons[image_path] = process_image(image_path, class_name, descriptors_amount)

    # Combine descriptors from all images in image_etalons into a single list
    descriptors_combined_etalons = [descriptor for descriptors_list in descriptors_by_image_etalons.values() for descriptor in descriptors_list]

    # Initialize class counts for image_input
    class_counts_by_image_input = {image_path: {class_name: 0 for _, class_name in image_etalons} for image_path, _ in image_input}

    # Compare descriptors from image_input with combined_descriptors from image_etalons
    for image_path, descriptors_list in descriptors_by_image_input.items():
        print(f'Comparing descriptors for image {image_path} ({len(descriptors_list)}) with combined set of unmarked descriptors... ({len(descriptors_combined_etalons)})')
        compare_descriptors(descriptors_list, descriptors_combined_etalons, class_counts_by_image_input[image_path])

    # Print class counts for image_input
    for image_path, class_counts in class_counts_by_image_input.items():
        print_class_counts(image_path, class_counts)

    #------------------------------I level---------------------------------------#

    print('---------------------------------I level--------------------------------------------')

    unmarked_descriptors_by_image_first_level = {}

    for image_path, descriptors_list in descriptors_by_image_etalons.items():
        for descriptor in descriptors_list:
            descriptor.mark_closest_descriptors(descriptors_list)
        
        marked_count = sum(1 for descriptor in descriptors_list if descriptor.marked)
        print(f"Для изображения {image_path}:")
        print("Количество отмеченных дескрипторов:", marked_count)
        print("Количество дескрипторов, которые остались:", len(descriptors_list) - marked_count)
        print()

        unmarked_descriptors_list = [descriptor for descriptor in descriptors_list if not descriptor.marked]
        unmarked_descriptors_by_image_first_level[image_path] = unmarked_descriptors_list

    for image_path, descriptors_list in unmarked_descriptors_by_image_first_level.items():
        print(f"Для изображения {image_path}:")
        print("Количество неотмеченных дескрипторов:", len(descriptors_list))
        print()

    descriptors_combined_etalons = [descriptor for descriptors_list in unmarked_descriptors_by_image_first_level.values() for descriptor in descriptors_list]

    class_counts_by_image_unmarked = {image_path: {class_name: 0 for _, class_name in image_etalons} for image_path, _ in image_input}

    for image_path, descriptors_list in descriptors_by_image_input.items():
        print(f'\nComparing descriptors for image {image_path} ({len(descriptors_list)}) with combined set of unmarked descriptors... ({len(descriptors_combined_etalons)})')
        compare_descriptors(descriptors_list, descriptors_combined_etalons, class_counts_by_image_unmarked[image_path])

    for image_path, class_counts in class_counts_by_image_unmarked.items():
        print_class_counts(image_path, class_counts)

    # #------------------------------II level---------------------------------------#

    print('---------------------------------II level--------------------------------------------')

    unmarked_descriptors_by_image_second_level = {}

    for image_path, descriptors_list in unmarked_descriptors_by_image_first_level.items():
        for descriptor in descriptors_list:
            descriptor.mark_closest_descriptors(descriptors_list)
        
        marked_count = sum(1 for descriptor in descriptors_list if descriptor.marked)
        print(f"Для изображения {image_path}:")
        print("Количество отмеченных дескрипторов:", marked_count)
        print("Количество дескрипторов, которые остались:", len(descriptors_list) - marked_count)

        unmarked_descriptors_list = [descriptor for descriptor in descriptors_list if not descriptor.marked]
        unmarked_descriptors_by_image_second_level[image_path] = unmarked_descriptors_list

    for image_path, descriptors_list in unmarked_descriptors_by_image_second_level.items():
        print(f"Для изображения {image_path}:")
        print("Количество неотмеченных дескрипторов:", len(descriptors_list))

    descriptors_combined_etalons = [descriptor for descriptors_list in unmarked_descriptors_by_image_second_level.values() for descriptor in descriptors_list]

    class_counts_by_image_unmarked = {image_path: {class_name: 0 for _, class_name in image_etalons} for image_path, _ in image_input}

    for image_path, descriptors_list in descriptors_by_image_input.items():
        print(f'\nComparing descriptors for image {image_path} ({len(descriptors_list)}) with combined set of unmarked descriptors... ({len(descriptors_combined_etalons)})')
        compare_descriptors(descriptors_list, descriptors_combined_etalons, class_counts_by_image_unmarked[image_path])

    for image_path, class_counts in class_counts_by_image_unmarked.items():
        print_class_counts(image_path, class_counts)

    
    # # #------------------------------III level---------------------------------------#

    print('---------------------------------III level--------------------------------------------')

    unmarked_descriptors_by_image_third_level = {}

    for image_path, descriptors_list in unmarked_descriptors_by_image_second_level.items():
        for descriptor in descriptors_list:
            descriptor.mark_closest_descriptors(descriptors_list)
        
        marked_count = sum(1 for descriptor in descriptors_list if descriptor.marked)
        print(f"Для изображения {image_path}:")
        print("Количество отмеченных дескрипторов:", marked_count)
        print("Количество дескрипторов, которые остались:", len(descriptors_list) - marked_count)

        unmarked_descriptors_list = [descriptor for descriptor in descriptors_list if not descriptor.marked]
        unmarked_descriptors_by_image_third_level[image_path] = unmarked_descriptors_list

    for image_path, descriptors_list in unmarked_descriptors_by_image_third_level.items():
        print(f"Для изображения {image_path}:")
        print("Количество неотмеченных дескрипторов:", len(descriptors_list))

    descriptors_combined_etalons = [descriptor for descriptors_list in unmarked_descriptors_by_image_third_level.values() for descriptor in descriptors_list]

    class_counts_by_image_unmarked = {image_path: {class_name: 0 for _, class_name in image_etalons} for image_path, _ in image_input}

    for image_path, descriptors_list in descriptors_by_image_input.items():
        print(f'\nComparing descriptors for image {image_path} ({len(descriptors_list)}) with combined set of unmarked descriptors... ({len(descriptors_combined_etalons)})')
        compare_descriptors(descriptors_list, descriptors_combined_etalons, class_counts_by_image_unmarked[image_path])

    for image_path, class_counts in class_counts_by_image_unmarked.items():
        print_class_counts(image_path, class_counts)


    # # #------------------------------IV level---------------------------------------#

    print('---------------------------------IV level--------------------------------------------')

    unmarked_descriptors_by_image_fourth_level = {}

    for image_path, descriptors_list in unmarked_descriptors_by_image_third_level.items():
        for descriptor in descriptors_list:
            descriptor.mark_closest_descriptors(descriptors_list)
        
        marked_count = sum(1 for descriptor in descriptors_list if descriptor.marked)
        print(f"Для изображения {image_path}:")
        print("Количество отмеченных дескрипторов:", marked_count)
        print("Количество дескрипторов, которые остались:", len(descriptors_list) - marked_count)

        unmarked_descriptors_list = [descriptor for descriptor in descriptors_list if not descriptor.marked]
        unmarked_descriptors_by_image_fourth_level[image_path] = unmarked_descriptors_list

    for image_path, descriptors_list in unmarked_descriptors_by_image_fourth_level.items():
        print(f"Для изображения {image_path}:")
        print("Количество неотмеченных дескрипторов:", len(descriptors_list))

    descriptors_combined_etalons = [descriptor for descriptors_list in unmarked_descriptors_by_image_fourth_level.values() for descriptor in descriptors_list]

    class_counts_by_image_unmarked = {image_path: {class_name: 0 for _, class_name in image_etalons} for image_path, _ in image_input}

    for image_path, descriptors_list in descriptors_by_image_input.items():
        print(f'\nComparing descriptors for image {image_path} ({len(descriptors_list)}) with combined set of unmarked descriptors... ({len(descriptors_combined_etalons)})')
        compare_descriptors(descriptors_list, descriptors_combined_etalons, class_counts_by_image_unmarked[image_path])

    for image_path, class_counts in class_counts_by_image_unmarked.items():
        print_class_counts(image_path, class_counts)


    # # #------------------------------V level---------------------------------------#

    print('---------------------------------V level--------------------------------------------')

    unmarked_descriptors_by_image_five_level = {}

    for image_path, descriptors_list in unmarked_descriptors_by_image_fourth_level.items():
        for descriptor in descriptors_list:
            descriptor.mark_closest_descriptors(descriptors_list)
        
        marked_count = sum(1 for descriptor in descriptors_list if descriptor.marked)
        print(f"Для изображения {image_path}:")
        print("Количество отмеченных дескрипторов:", marked_count)
        print("Количество дескрипторов, которые остались:", len(descriptors_list) - marked_count)

        unmarked_descriptors_list = [descriptor for descriptor in descriptors_list if not descriptor.marked]
        unmarked_descriptors_by_image_five_level[image_path] = unmarked_descriptors_list

    for image_path, descriptors_list in unmarked_descriptors_by_image_five_level.items():
        print(f"Для изображения {image_path}:")
        print("Количество неотмеченных дескрипторов:", len(descriptors_list))

    descriptors_combined_etalons = [descriptor for descriptors_list in unmarked_descriptors_by_image_five_level.values() for descriptor in descriptors_list]

    class_counts_by_image_unmarked = {image_path: {class_name: 0 for _, class_name in image_etalons} for image_path, _ in image_input}

    for image_path, descriptors_list in descriptors_by_image_input.items():
        print(f'\nComparing descriptors for image {image_path} ({len(descriptors_list)}) with combined set of unmarked descriptors... ({len(descriptors_combined_etalons)})')
        compare_descriptors(descriptors_list, descriptors_combined_etalons, class_counts_by_image_unmarked[image_path])

    for image_path, class_counts in class_counts_by_image_unmarked.items():
        print_class_counts(image_path, class_counts)

















