from scipy.spatial import distance


class Descriptor:
    def __init__(self, descriptor: list, marked: bool, index: int):
        self.descriptor = descriptor
        self.marked = marked
        self.index = index

    def hamming_distance(self, other_descriptor):
        # print(f'{len(self.descriptor)} | {len(other_descriptor.descriptor)}')
        return float(len(self.descriptor)) * distance.hamming(self.descriptor, other_descriptor.descriptor)

    @staticmethod
    def get_threshold(descriptor_length):
        # return descriptor_length * 0.375
        return descriptor_length * 0.5

    def mark_closest_descriptors(self, descriptors_list):
        threshold = self.get_threshold(len(self.descriptor))
        min_distance = float('inf')
        min_distance_bigger_then_threshold = float('inf')
        closest_descriptor = None
        for i in range(self.index + 1, len(descriptors_list)):
            other_descriptor = descriptors_list[i]
            if self.marked:
                continue
            if other_descriptor.marked:
                # print(f'{other_descriptor.index} was marked')
                continue  # Skip marked descriptors
            try:
                dist = self.hamming_distance(other_descriptor)
                # print(f'Comparing {self.index} and {other_descriptor.index}: {dist}')
            except ValueError:
                continue  # Skip descriptors with different lengths
            if dist < min_distance and dist < threshold:
                min_distance = dist
                closest_descriptor = other_descriptor
            if dist < min_distance_bigger_then_threshold:
                min_distance_bigger_then_threshold = dist

        # print('\n')

        # if self.marked:
        #     print(f'{self.index} was already marked as closest')
        # else:
        #     print(f'min distance: {min_distance_bigger_then_threshold}')

        if closest_descriptor is not None:
            closest_descriptor.marked = True
        #     print(f'min distance < {threshold}: {min_distance}')
        #     print(f'closest {closest_descriptor.index}')
        #     print(f'{closest_descriptor.index} was marked as closest')
        # print('-------------------------')

