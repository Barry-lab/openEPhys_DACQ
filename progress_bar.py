# -*- coding: utf-8 -*-
import sys
# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=40, initiation=False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    if not initiation:
        sys.stdout.flush()
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('%s |%s| %s%s %s\r' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

# # 
# # Sample Usage
# # 

# from time import sleep

# # A List of Items
# items = list(range(0, 57))
# l = len(items)
# # Initial call to print 0% progress
# print_progress(0, l, prefix = 'Progress:', suffix = 'Complete',initiation=True)
# for i, item in enumerate(items):
#     # Do stuff...
#     sleep(0.1)
#     # Update Progress Bar
#     print_progress(i + 1, l, prefix = 'Progress:', suffix = 'Complete')

