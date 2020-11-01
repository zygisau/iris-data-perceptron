def split_files(file, indicator_1, indicator_2, file_name_1,
                file_name_2, class_name_1, class_name_2):
    output_1 = open(file_name_1, 'w')
    output_2 = open(file_name_2, 'w')

    line = file.readline()
    while line:
        words = line.split(',')
        if any(pattern in line for pattern in indicator_1):
            words[-1] = str(class_name_1)
            output_1.write(','.join(words))
            output_1.write('\n')
        if any(pattern in line for pattern in indicator_2):
            words[-1] = str(class_name_2)
            output_2.write(','.join(words))
            output_2.write('\n')
        line = file.readline()

    output_1.close()
    output_2.close()


def split_into_first_set(file):
    OUTPUT_1_FILE = 'data/dataset_1_1.txt'
    OUTPUT_2_FILE = 'data/dataset_1_2.txt'
    OUTPUT_1_INDICATOR = ['setosa']
    OUTPUT_2_INDICATOR = ['versicolor', 'virginica']
    OUTPUT_1_CLASS = 0
    OUTPUT_2_CLASS = 1
    split_files(file, OUTPUT_1_INDICATOR, OUTPUT_2_INDICATOR, OUTPUT_1_FILE,
                OUTPUT_2_FILE, OUTPUT_1_CLASS, OUTPUT_2_CLASS)


def split_into_second_set(file):
    OUTPUT_1_FILE = 'data/dataset_2_1.txt'
    OUTPUT_2_FILE = 'data/dataset_2_2.txt'
    OUTPUT_1_INDICATOR = ['versicolor']
    OUTPUT_2_INDICATOR = ['virginica']
    OUTPUT_1_CLASS = 0
    OUTPUT_2_CLASS = 1
    split_files(file, OUTPUT_1_INDICATOR, OUTPUT_2_INDICATOR, OUTPUT_1_FILE,
                OUTPUT_2_FILE, OUTPUT_1_CLASS, OUTPUT_2_CLASS)


if __name__ == '__main__':
    file_name = "data/iris.data"
    with open(file_name) as f:
        split_into_first_set(f)
    with open(file_name) as f:
        split_into_second_set(f)
