from enum import Enum

class DataSetType(Enum):
    FIRST_SET = ['data/dataset_1_1.txt',
                'data/dataset_1_2.txt']
    SECOND_SET = ['data/dataset_2_1.txt',
                'data/dataset_2_2.txt']


class FetchingService:
    
    @staticmethod
    def __read_file(file_name):
        contents = []
        with open(file_name) as f:
            line = f.readline()
            while line:
                parsed_line = [float(value.rstrip()) for value in line.split(',')]
                contents.append(parsed_line)
                line = f.readline()
        return contents
        
    @staticmethod
    def __split_content(data_set):
        return [data_set[:int(len(data_set)*0.8)], data_set[-int(len(data_set)*0.2):]]
    
    @staticmethod
    def fetch_data(set_type):
        set_a = FetchingService.__read_file(set_type.value[0])
        set_b = FetchingService.__read_file(set_type.value[1])
        training_a, testing_a = FetchingService.__split_content(set_a)
        training_b, testing_b = FetchingService.__split_content(set_b)
        return [(training_a + training_b), (testing_a + testing_b)]