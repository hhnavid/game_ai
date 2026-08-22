import numpy as np
import csv
from csv import DictWriter


class CSVLogger(object):

    def __init__(self, csv_file_path, keys, mode):
        """
        :param csv_file_path: the path in which the file will be saved
        :param keys: (list) headers for the csv file
        :param mode: (str) Logger mode; if set to 'write', a new csv file is created to store data. If set to 'read',
        content of an existing csv file is loaded.
        https://www.tutorialspoint.com/How-to-save-a-Python-Dictionary-to-CSV-file
        """
        self.save_path = csv_file_path
        self.csv_columns = keys
        self.mode = mode
        if self.mode == 'write':
            # Create csv with given headers
            try:
                with open(self.save_path, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.csv_columns)
                    writer.writeheader()
            except IOError:
                print("I/O error")

    def append_dict_as_row(self, dict_of_elem, field_names):
        """
        :param dict_of_elem: dict. to be appended to csv file
        :param field_names: dict. keys
        https://thispointer.com/python-how-to-append-a-new-row-to-an-existing-csv-file/
        """
        if self.mode == 'write':
            # Open file in append mode
            with open(self.save_path, 'a+', newline='') as write_obj:
                dict_writer = DictWriter(write_obj,
                                         fieldnames=field_names)  # Create a writer object from csv module
                dict_writer.writerow(dict_of_elem)                # Add dictionary as row in the csv
        else:
            raise Exception("Can't append data in read mode!")

    def csv2dict(self):
        """
        Reads the csv file and returns its corresponding dictionary. 1st row of the file is assumed as
        dictionary key values
        :param file_name: path to csv file
        :return: python dict.
        https://overlaid.net/2016/02/04/convert-a-csv-to-a-dictionary-in-python/
        """
        csv_dict = {}
        with open(self.save_path, mode='r') as infile:
            reader = csv.reader(infile)
            for idx, row in enumerate(reader):
                if idx == 0:
                    # Extract field names and use as dict.keys
                    for key in row:
                        csv_dict[key] = []
                else:
                    for key, value in zip(csv_dict.keys(), row):
                        if type(value) is not list:
                            value = [value]
                        csv_dict[key] += list(map(float, value))  # Convert values from str to float and append
        return csv_dict


def main():
    # Create logger
    csv_file_name = "../tests/Names.csv"
    # logger = CSVLogger(csv_file_name, keys=['No*', 'Name*', 'Country*'], mode='write')
    #
    # # Append rows
    # data_row = {'No': 6, 'Name': 'Ajax', 'Country': 'Iraq'}
    # logger.append_dict_as_row(data_row, data_row.keys())
    # data_row = {'No': 16, 'Name': 'bb', 'Country': 'uk1'}
    # logger.append_dict_as_row(data_row, data_row.keys())
    # data_row = {'No': 17, 'Name': 'max', 'Country': 'swd'}
    # logger.append_dict_as_row(data_row, data_row.keys())

    # Load data
    logger = CSVLogger(csv_file_name, keys=['No*', 'Name*', 'Country*'], mode='read')
    dict_ = logger.csv2dict()
    print(dict_)


if __name__=='__main__':
    main()