import numpy as np
import sys

gSkippedTags = ['pyaf.timing', '_TIME_IN_SECONDS' , "PYAF_SYSTEM_DEPENDENT_",
                "_START", "_END", "system_uname", "EXECUTION_TIME_DETAIL",
                "CreationDate" , "Training_Time" , "matplotlib.font_manager"]


if(len(sys.argv) != 3):
    print("NUM_DIFF_ERROR_INVALID_ARGS" , sys.argv);
    sys.exit(-1)

def is_numeric(x):
    try:
        val = float(x)
    except ValueError:
        return False;
    return True;

# some warnings contain hexadecimal address which may change from a run to another.
# Example : _matplotlib/core.py:807: UserWarning: The label '___OC_Forecast' of <matplotlib.lines.Line2D object at 0x7f571157cd50> starts with '_'. It is thus excluded from the legend.
def is_hex_address(x):
    return x.startswith('0x')

# the goal here is to compare the two words inside a JSON and
# allow some small numeric differences
# (used to compare dataframe.to_json() in the logs).

def compare_words(word_orig, word_new):
    if(word_orig == word_new):
        return 0;
    if(is_hex_address(word_orig) and is_hex_address(word_new)):
        return 0;
    if(is_numeric(word_orig) and is_numeric(word_new)):
        lNumber_orig = float(word_orig);
        lNumber_new = float(word_new);
        if(np.isclose([lNumber_orig] , [lNumber_new])):
            lRelDiff = abs(lNumber_new - lNumber_orig) / (abs(lNumber_orig) + 1e-10);
            if(lRelDiff > 1e-12):
                print("NUM_DIFF_DEBUG_ALLOWED_SMALL_DIFFERENCE", word_orig, word_new, lNumber_orig, lNumber_new, lRelDiff)
            return 0;
        else:
            print("DIFFERENT_WORDS" , word_orig, word_new)    
            return 1;
    else:
        # print("NUM_DIFF_NOT_NUMERIC_DIFF", word_orig, word_new)        
        return 1;
    



def compare_lines(line_orig, line_new):
    if(line_orig == line_new):
        return 0;

    import re
    lRegex  = '[\[?,:"{}\]\= \n\t]'
    split_orig = re.split(lRegex, line_orig)
    split_new = re.split(lRegex, line_new)
    # print("SPLIT_ORIG", split_orig)
    # print("SPLIT_NEW", split_new)

    N_orig = len(split_orig)
    N_new = len(split_new)
    if(N_orig != N_new):
        print("NUM_DIFF_LINE_WITH_DIFFERENT_NUMBER_OF_WORDS" , N_orig, N_new);
        return 1;

    N = min(N_orig, N_new);
    # compare the first N words of the two lines.
    out = 0;
    for l in range(N):
        result = compare_words(split_orig[l] , split_new[l])
        out = out + result;
    
    return out;

def compare_files(file_orig, file_new):
    import os.path
    if(not os.path.exists(file_orig)):
        print("ORIGINAL_FILE_DOES_NOT_EXIST" , file_orig)
        print("EXEC_THIS=1 cp ", file_new, file_orig)
        return
    
    with open(file_orig) as f:
        content_orig = f.readlines()
        for tag in gSkippedTags:
            content_orig = [x for x in content_orig if tag not in x]

    with open(file_new) as f:
        content_new = f.readlines()
        for tag in gSkippedTags:
            content_new = [x for x in content_new if tag not in x]

    N_orig = len(content_orig)
    N_new = len(content_new)
    if(N_orig != N_new):
        print("NUM_DIFF_FILES_WITH_DIFFERENT_NUMBER_OF_LINES" , N_orig, N_new);

    lDiff = abs(N_orig - N_new)
    if(lDiff > 20):
       print("NUM_DIFF_FILES_WITH_TOO_MUCH_DIFF" , N_orig, N_new);
       print("NUM_DIFF_NUMBER_OF_DIFFERENT_LINES_AT_LEAST" , lDiff)
       print("EXEC_THIS=1 cp ", file_new, file_orig)
       print("NUM_DIFF_FILES_ARE_DIFFERENT")
       return;
        
    N = min(N_orig, N_new);
    # compare the first N lines of the two files.
    out = 0;
    diffs = [];
    for l in range(N):
        result = compare_lines(content_orig[l] , content_new[l])
        if(result > 0):
            diffs.append((content_orig[l] , content_new[l]));
        out = out + result;

    if(out > 0):
        print("NUM_DIFF_NUMBER_OF__DIFFERENT_LINES" , out)
        print("FIRST_DIFFERENCE_ORIG" , diffs[0][0])
        print("FIRST_DIFFERENCE_NEW" , diffs[0][1])
        print("LAST_DIFFERENCE_ORIG" , diffs[-1][0])
        print("LAST_DIFFERENCE_NEW" , diffs[-1][1])
        print("EXEC_THIS=1 cp ", file_new, file_orig)
        print("NUM_DIFF_FILES_ARE_DIFFERENT")


file_orig = sys.argv[1]
file_new = sys.argv[2]

try:
    compare_files(file_orig, file_new);
except Exception as e:
    lStr = str(type(e).__name__) + " " + str(e)
    print("FILE_COMPARISON_FAILED_WITH_ERROR" , lStr);
