
import os
import sys


def make_filename(report_folder,folder,filename,should_exist = False):
    if report_folder is None:
        return None
    else:
        if not folder is None:
            report_folder = os.path.join(report_folder,'exploratory')
        if not os.path.exists(report_folder):
            os.makedirs(report_folder)
        filename = os.path.join(report_folder,filename)
        if should_exist and not os.path.isfile(filename):
            print("File '" + filename + "' does not exist.")
            sys.exit(1)

        return filename
