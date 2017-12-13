import os

DATA_PATH = '/Users/sai/dev/datasets/netflix/'
FILES = ['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt']
COMBINED_RATINGS = 'combined_ratings.csv'


def combine_to_csv():
    csv_lines = list()
    csv_lines.append('movieId,userId,rating,timestamp\n')
    for fname in FILES:
        path = os.path.join(DATA_PATH, fname)
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.endswith(':\n'):
                movie_id = line.split(':')[0]
                continue
            new_line = ','.join([movie_id, line])
            csv_lines.append(new_line)
    with open(os.path.join(DATA_PATH, COMBINED_RATINGS), 'w') as f:
        for item in csv_lines:
            f.write(item)


if __name__ == "__main__":
    combine_to_csv()
