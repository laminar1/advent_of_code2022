#!/usr/bin/env python
#
# execute_day.py
#
# Created by Scott Lee on 11/30/22.
# Copyright (c) 2022 Apple, inc. All rights reserved.
#

'''
Description:

'''

import numpy as np

def day1():
    print('hello world')
    f = open("day1_data.txt", "r")
    # print(f.readlines())

    calories_list = []

    elf_index = 1
    current_calories = 0
    for line in f.readlines():
        if line == '\n':
            calories_list.append(current_calories)
            current_calories = 0
        else:
            current_calories += int(line.replace('\n',''))

    print(calories_list)
    print(np.argmax(calories_list) )

def day2():
    f = open("day2_data.txt", "r")

    total_score = 0
    for line in f.readlines():
        # get line item score
        if 'X' in line:
            total_score += 1
            if 'A' in line:
                total_score += 3 # draw
            elif 'B' in line:
                total_score += 0 # lose
            elif 'C' in line:
                total_score += 6  # win
        elif 'Y' in line:
            total_score += 2
            if 'A' in line:
                total_score += 6 # draw
            elif 'B' in line:
                total_score += 3 # win
            elif 'C' in line:
                total_score += 0 # loss
        elif 'Z' in line:
            total_score += 3
            if 'A' in line:
                total_score += 0 # draw
            elif 'B' in line:
                total_score += 6 # win
            elif 'C' in line:
                total_score += 3  # loss

    print(f'part 1: {total_score}')
    f.close()

    f = open("day2_data.txt", "r")
    total_score = 0
    for line in f.readlines():
        if 'X' in line:
            total_score += 0
            if 'A' in line:
                total_score += 3  #
            elif 'B' in line:
                total_score += 1  # lose
            elif 'C' in line:
                total_score += 2  # win
        elif 'Y' in line:
            total_score += 3
            if 'A' in line:
                total_score += 1  # draw
            elif 'B' in line:
                total_score += 2  # win
            elif 'C' in line:
                total_score += 3  # loss
        elif 'Z' in line:
            total_score += 6
            if 'A' in line:
                total_score += 2  # draw
            elif 'B' in line:
                total_score += 3  # win
            elif 'C' in line:
                total_score += 1  # loss

    print(f'part 2: {total_score}')
    f.close()




        # get win/tie/loss score

def day3():
    print('day 3 fun')
    f = open("day3_data.txt", "r")
    # f = open("day3_debug.txt", "r")
    total_score = 0
    for line in f.readlines():
        # separate into compartments
        line = line.replace('\n','')
        # print(len(line.replace('\n','')))
        c1 = line[:int(len(line)/2)]
        c2 = line[-int(len(line)/2):]

        common = [x for x in c1 if x in c2]
        common = set(common)
        common = list(common)

        letter_list = ['abcdefghijklmnopqrstuvwxyz']
        upper_list = [x.upper() for x in letter_list]
        all_priority = letter_list[0] + upper_list[0]

        line_score = 0
        # print(common)
        for k in common:
            line_score += 1 + all_priority.index(k)

        # print(f'line_score: {line_score}')
        total_score += line_score


    print(f'part 1: total score {total_score}')
    f.close()

    with open('day3_data.txt') as f:
        lines = f.read().splitlines()
    total_score = 0
    for k in range(0,len(lines),3):
        print(k)
        e1 = lines[k].replace('\n','')
        e2 = lines[k+1].replace('\n', '')
        e3 = lines[k+2].replace('\n', '')

        common_e1_e2 = [x for x in e1 if x in e2]
        common_all = [x for x in e3 if x in common_e1_e2]

        common_all = set(common_all)
        common_all = list(common_all)

        for k in common_all:
            line_score = 1 + all_priority.index(k)

        print(f'line_score: {line_score}')
        total_score += line_score

        print(common_all)

    print(f'part 2: total score {total_score}')
    
    # break

def day4():
    print('hello world')
    f = open("day4.txt", "r")
    overlapping=0
    for line in f.readlines():
        line = line.replace('\n','')
        e1 = line.split(',')[0]
        e2 = line.split(',')[-1]
        e1_1 = int(e1.split('-')[0])
        e1_2 = int(e1.split('-')[-1])
        e2_1 = int(e2.split('-')[0])
        e2_2 = int(e2.split('-')[-1])

        if (e1_1 <= e2_1) & (e1_2 >= e2_2):
            overlapping += 1
        elif (e2_1 <= e1_1) & (e2_2 >= e1_2):
            overlapping += 1
    f.close()

    print(f'part 1: overlapping: {overlapping}')
    f = open("day4.txt", "r")
    overlapping = 0
    for line in f.readlines():
        line = line.replace('\n','')
        print(line)
        e1 = line.split(',')[0]
        e2 = line.split(',')[-1]
        e1_1 = int(e1.split('-')[0])
        e1_2 = int(e1.split('-')[-1])
        e2_1 = int(e2.split('-')[0])
        e2_2 = int(e2.split('-')[-1])

        if (e1_1 >= e2_1) & (e1_1 <= e2_2):
            overlapping += 1
        elif (e1_2 >= e2_1) & (e1_2 <= e2_2):
            overlapping += 1
        elif (e2_1 >= e1_1) & (e2_1 <= e1_2):
            overlapping += 1
        elif (e2_2 >= e1_1) & (e2_2 <= e1_2):
            overlapping += 1


        # if (e1_1 <= e2_1) or (e1_2 >= e2_2):
        #     print('overlapped')
        #     overlapping += 1
        # elif (e2_1 <= e1_1) or (e2_2 >= e1_2):
        #     print('overlapped')
        #
        #     overlapping += 1
    f.close()

    print(f'part 2: overlapping: {overlapping}')

    # f = open("day4.txt", "r")
    # print(f.readlines())
    return

def day5():
    import re, copy
    print('hello world')
    # f = open("day5_debug.txt", "r")
    # f = open("day5.txt", "r")
    with open('day5.txt') as f:
        lines = f.read().splitlines()
    position = {}
    for line in lines:
        # initial input
        if line[0:3] == ' 1 ':
            break
        indices = [m.start() for m in re.finditer('\[', line)]
        for idx in indices:
            column = int(idx/4) + 1
            # print(f'column: {column}')
            if column not in position:
                position[column] = []
            position[column].append(line[idx+1])
    # reverse order
    for key in position:
        position[key].reverse()
    print(position)

    original_position = copy.deepcopy(position)

    # move commands
    for line in lines:
        if line != '' and line[0] =='m':
            num_crates = int(line.split(' ')[1])
            stack0 = int(line.split(' ')[3])
            stack1 = int(line.split(' ')[5])

            for k in range(num_crates):
                position[stack1].append(position[stack0].pop())

    print(position)
    keys = list(position.keys())
    keys.sort()

    output = []
    for k in range(max(keys)):
        output.append(position[k+1][-1])

    print(f'output: {"".join(output)}')

    ## part 2
    position = copy.deepcopy(original_position)
    print('part 2')

    # move commands
    for line in lines:
        if line != '' and line[0] =='m':
            print(line)
            num_crates = int(line.split(' ')[1])
            stack0 = int(line.split(' ')[3])
            stack1 = int(line.split(' ')[5])

            # for k in range(num_crates):
            print('before')
            print(position)
            position[stack1] += position[stack0][-num_crates:]
            position[stack0] = position[stack0][:-num_crates]
            print('after')
            print(position)

    print(position)
    keys = list(position.keys())
    keys.sort()

    output = []
    for k in range(max(keys)):
        output.append(position[k+1][-1])

    print(f'output: {"".join(output)}')

def day6():
    day6_txt = 'hjchjcjhjshjsssrfsrrldrddbrrzfzjzzrffvwwclwlffhwhwpwfffcbctbccchmccfmmwdmmdwwdttwffsfshswhhfchfchcphpnnflnlznlnnnvpnnhjhrrjgrglgwwrgghzhnzztlltbtwbbvmmzppdmmhchnccspccvmvwwzpwzzmddjmdmbdmmrzmmhlhhhdndllrlgrllmlhljlmmgdmggwdggffblbmmdgdwgdgwgvgcgtctjjnfnsffbqbwbnnsbbqsswggncgncntccqqfmmlqllllvrvlrlgldlggjvvqdvqvzvzpzrrvfvcfcqfcflfjfjrrwbrrpnrrvzzbddfdgfdgfddppbdbbjddtcdtccllvccjtctmccsttfcfppmvppbvpvjpvprvvmggjffbqbbqhbbcdbdrbrjrllgmggwdwzddzdczddgpgglgpgqgmgllrqlqmlqllmbmzmdzdbzddqbbzwwjfjqqlrrzgrrlzzdczddlflpfpqfqrqzzpdzdnznwwvjjnndldpdppfgppgwwnddmzzmffvgggphpbpnbblqbqccswwlcwlltvvlggqvqhvvtstmssbvvflvvhdhggqpgqgqlqggjvjpphhqnhqhvqhvqvpvvfvqfqvvbmbtmmqpqccwcbbqwbbmjbjsscsqqcccbjbvvmsmrmwrwwbhwwdpwphprptrrdssrprjrdrssqtqzqtzzcbbvrrpwrwlwbbpwwrddfcfccvttdsshqhqddsmslsffdsdrrswsmmztzhzghgqhqbbfcctmcmmdwwpbpdbbnjbnbmnmqqqftfdfnfzfqzffmmnqqgfgwwntnwwfsfmssnzzscsmmzttdwttfppnccngggmrmbmccrbcrclcslsfspfsswnwpwjwddbnbjbrjbrjbrjbrrlgljggcpgptpltlmlnnjpnptnppdcdwcwfcwcnnzddbrrnbnnsjsnjjjsqjjqhjjbrjbbghbgggsdshswsrwwqbwqqmtqmqdmdttwfwzztwztwtfwtwtfwwjpwjjljcccdbdndmmhjhzzszfssgbbwbhhhrhfrhhwttwltwtbwbrbqbpbwwjtwwdjjwhwmmddhgdhhhdrhrjjngjgnjnfjnjrjtrrhmmjzzsjzjhjmmvnmnqqzbbtnncffnhnhgnhggcssvrrwfwjwgjwwwdtwddgrrwnwffndnbdnbnsnbsnndtdvvdtdsdhshcczqzztzfzdzndnnhqqzggwrwtrtvtjvjvsjjzhzggtffdbbzwbbnwwnhnpnznfncnqqvmvbmbcbgcgfggtpggjqggnppwmwzzqpzzvvgqqjjhrrmmfsfvvbqvbqqptpztpzzhnndgngdngdngnllslhhsvssdffwllchlccjpjdpddgcgrgttmqttjlttphhdwdtdtnnrggmhhmhmgghnhbhppfmmlrmmlqqsllwswpsshbbfjjqvvlsvvblbljltjtjsszcscmsmgmddzsdddznnvvddwbwjwwmqwqlqvqffptfpttgngpgttlglpgppssthshwhzhshfshssstppbpjjtgjttlsszmzccrllwjjcdccgmccdppnlnrlltntcnttrhtrtqqfhhjfffldfdldbbqjqzzqjzzfhhtvvrrfqfsqfftrtbblmlsltsszqqcsshllvlbbpgpzgpgcppwnpwnwpnnjjmllrhllfzlzglzlnlrrcssjjjtgjgjjznjnmmjjpccqrqzrzzmfzfczcssnttddfjdfjfpjjmqjqnqtqbqzzqqzwwzwqqfccvcrvrqrlqrlqqnvjtqswzvngfcjpmnrnvnwtwnjvsmzhtwzpjbpglchwfvwhvznsvhvwwjppmqqpcpmzrznqrlvbgdfcpgdtfhwdclvzjqlhtbdvsgpjlrgbcjblnqhffbcjfwsgssfzlsbhrptfgsfsstzbwqcsrpgftblrnldhwfwpgpffftsjgclzqmjmcvwjrsbhgdblswrwnhpjtgsggmnjqgzzctjjztwhcqvhqfvddljjtqwgpmwdsmmhdttvdpqpvsqbpwmtzgfthtmfhmplmwqcbmdmrwqmmzmjmfdbqspmshlhtbmbcpcjsgdccwmbfwvftlshtrgzvbndqvqjzqgbgrnmzbwfgntfphjrvhrgzgdqclvpvwffghthlqwlghfqrwpdmgnthqwznqsjrnnpghfcfwctpvnbnftczlhmdslfvqprhgqmzmzsjvtzfsfzlcrltjhfgmwqcvnzmttfvvbsjslqfwmnhgbbjdwfgbzsjqfsgvphvmclfgtmcvlpslpqfsbzgccqslmrgdwrtlrzbbvrjrnnnncgrnsggzjrfqtmhjdvdfbwdmqrjbghrbnhpqcdzgbqwrvrcpwdlbvrdpfhnpbncjzgmftjhvwnplmnlfnlfjsjnqhtgqldzqlrlqtdjndjpsfdcdfrwtqblzpsqjvnqchdhwvswrmczhsbpfggsvzdznqjlrjjbcjnsjvqtrtttmmcgdwbqcthqvzffjmdbvjmjvcrmnpjgtjshbnlqpdfdnbcfmbrzsvqftrnfzmjdhpprpnwqbngmbbwjvmdzbwvttncdtgqnwwchmbbdtrwlflmqnbthnczfpmtpfpqpbwbcpsplgfpjfptdpvzjnbgzrfdwpdrztqtsrzmbfqhgwnfzcsbdsjsmbdghjcjlvbpjpplgqqnbqpqgsqqbmpgmlghrlbcfzjhlqfgdpfljspwqjbsjqzwwhrcpfrhwpvgrjpqjzfzphcrwbfwsjdsjtlwctsfhrmbnvsrfwwvgqtjtjvvqzjznlrsblgthjfrphsfmbtpmbthwdhrqbdmbzplbvpbcwvhgrsjccsnhbrqdbljzpdttbffqrbmgmzsmhdhsmnnmjqtwdjpmhlpwtwhvbfnjzfcwfzfplsbqgcvwgjcbwzbzmqdchwrggjwgjbttsttsztrftttqpslwvtcrjmdtwdhlwnhpjstqnqtvrmlmtcgjljzrthgpmvdjzlwfntqmbpdpgmmvvwqmdqqwrnlsrmhrpdtmhjrngwdfgddlrmdnfdnscjhdfjzwljjrsclnfdmhpbsvwtsmrdsvmpbjjpmmgtfclcccmcnzcslsvdwncrgtpvgbcgwdmcqlthgmrqmpnprlsqgzzpzzmhgflfgpjwgjjdpvggmhcstrwscqggqgrrjwtqzdfbnwgtvslvghfnzphbznqslcwwcsplgwjnltrttqzcvjhfdrwgjpclzmqfgvhzhrcdsgchhfqptqjwffmrsjrplzzlhnptwlmrvtstsrgnfdnrbtdbzjdcbthhtnjdprrpgtgfjsqnpgslcqgmdfpsrdfvhbqvvpthmmshpdnrrwlfcmqfbrsvqdqhffgbdhwjgcjsclcqpwnrfzfdqcvnmqnjmjnvhqmznmbnbnnjfrtlvbpdgglqpgcmqqcnpzfvvsnchpbjprpnwbdqvqzgjvgtnsrvswfmwhzllmlgpsglssnhcvbjtfghhrznpzntwwtnshmhhddnntdljhhhpmnchssqthbzpqmtmjbcfvmgnmwhpzrbwzzvzmnfdcsbvzphlglbhjpfmrtgfblhtszqvbbmtglwdhgjdvvgtpscgvwzjppfnlndnmtrnnnlfbgmrpqlvhvbgzmwghnsmdmdrftqpqncsbcmqhhhljzlwcrlsdbhrlddwlhcghvttjfsmfdzcllswjgsmcmghbflbdgpwfqplqnrvzfnctdsnmldhtbtpfrsztjdsgmnbrdjwbrgqlhdrlrnmlpwltgpwhwztbwpcqtwbqdmsfdfczftncvsggshhcqbjgcwjljcqdpczrnzbjhrhwcgrbbqzmmfjpqwrwppmnvcsfwprjqvtnzqzwtwlvvqssfjzbrvjjrmphtbjbrzttmvvhdfsnqdmpfbtprbqgzdgtjtpvbqqsgppsrnvsfnmgvbbsjcpttffthpvfjpnzmsjmpdzbldggtjrjqpshtmgpfgtcstdrgjhzjr'
    day6_debug = 'bvwbjplbgvbhsrlpgdmjqwftvncz'

    text_input = day6_txt
    for k in range(len(text_input) - 3):
        substring = list(text_input[k:(k+4)])
        print(substring)
        if len(np.unique(substring))==4:
            marker_start = k + 4
            break
    print(f'marker_start: {marker_start}')

    text_input = day6_txt
    for k in range(len(text_input) - 13):
        substring = list(text_input[k:(k+14)])
        print(substring)
        if len(np.unique(substring))==14:
            marker_start = k + 14
            break
    print(f'marker_start: {marker_start}')

    # print('hello world')
    # # f = open("day5_debug.txt", "r")
    # # f = open("day5.txt", "r")
    # with open('day5.txt') as f:
    #     lines = f.read().splitlines()
    # for line in lines:

def day7():
    import os
    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    print('hello')

    input = 'day7_data.txt'
    with open(input) as f:
        lines = f.read().splitlines()

    ls_output = False
    dir_struct = {'/':{}}
    pwd = '/'

    def find_current_dir(dir_struct, path):
        current_dir = dir_struct['/']
        if path == '/':
            return current_dir
        for item in path.split(os.sep):
            if item != '':
                current_dir = current_dir.get(item)
        return current_dir

    def find_directory_size(dir_or_filename):
        if isinstance(dir_or_filename, int):
            return dir_or_filename
        else:
            temp_total = 0
            for key in dir_or_filename:
                temp_total += find_directory_size(dir_or_filename[key])
            return temp_total

    for idx, line in enumerate(lines):
        if line.startswith('$ ls'):
            current_dir = find_current_dir(dir_struct, pwd)
            ls_output = True
            continue
        if line.startswith('$'):
            ls_output = False
        if ls_output:
            if line.startswith('dir'):
                dirname = line.split(' ')[-1]
                current_dir[dirname] = {}
            else:
                filesize = int(line.split(' ')[0])
                filename = line.split(' ')[-1]
                current_dir[filename] = filesize
            continue

        if line.startswith('$ cd /'):
            pwd = '/'
        elif line.startswith('$ cd ..'):
            pwd = pwd.rsplit(os.sep,1)[0]
        elif line.startswith('$ cd '):
            dirname = line.split(' ')[-1]
            if pwd == '/':
                pwd = pwd + dirname
            else:
                pwd = pwd + os.sep + dirname



    pp.pprint(dir_struct)

    def find_directory_sizes_recursive(dir_or_file):
        if isinstance(dir_or_file, int):
            return []
        else:
            dir_sizes = [find_directory_size(dir_or_file)]
            for key in dir_or_file:
                dir_sizes += find_directory_sizes_recursive(dir_or_file[key])
            return dir_sizes

    # find all directory sizes

    # print(find_directory_size(dir_struct))
    # print(find_directory_sizes_recursive(dir_struct))

    dir_sizes = np.array(find_directory_sizes_recursive(dir_struct))
    dir_sizes = dir_sizes[1:] # for some reason total dir size is repeated twice
    print(dir_sizes)
    small_dirs = dir_sizes[np.where(dir_sizes<100000)]
    print(small_dirs)
    print(f'part 1: sum of small dirs: {np.sum(small_dirs)}')

    # part 2
    total_memory = 70000000
    required_memory = 30000000

    available_memory = total_memory - dir_sizes[0]
    minimum_dirsize_to_delete = required_memory - available_memory
    dir_sizes_sorted = np.sort(dir_sizes)



    print(dir_sizes_sorted)
    print(minimum_dirsize_to_delete)

    print('part 2 answer')
    print(dir_sizes_sorted[np.where((dir_sizes_sorted - minimum_dirsize_to_delete)>0)][0])

def day8():
    print('hello')
    # fname = 'day8_debug.txt'
    fname = 'day8.txt'
    tree_map = np.genfromtxt(fname,delimiter=1).astype(int)
    is_visible = np.zeros(np.shape(tree_map))

    # all perimeter trees are visible
    is_visible[0,:] = 1
    is_visible[-1, :] = 1
    is_visible[:,0] = 1
    is_visible[:, -1] = 1

    def update_visible_map(start_range,end_range):
        # top to bottom
        is_visible = np.zeros(np.shape(tree_map))
        is_visible[0, :] = 1
        max_row = tree_map[0, :]
        for k in range(start_range, end_range):
            row_mask = tree_map[k, :] > max_row
            row_mask = [int(x) for x in row_mask]

            # update is_visible
            is_visible[k, :] = [a or b for a, b in zip(row_mask, is_visible[k, :])]

            max_row = np.maximum(tree_map[k, :],max_row)
            # print(max_row)
        return is_visible

    def update_visible_map_c(start_range, end_range):
        is_visible = np.zeros(np.shape(tree_map))
        is_visible[:, 0] = 1
        max_col = tree_map[:,0]
        for k in range(start_range, end_range):
            col_mask = tree_map[:, k] > max_col
            col_mask = [int(x) for x in col_mask]
            # check that adjacent tree is also visible
            # col_mask = [a and b for a, b in zip(col_mask, is_visible[:, k - 1])]

            # update is_visible
            is_visible[:,k] = [a or b for a, b in zip(col_mask, is_visible[:, k])]

            #update max_col
            max_col = np.maximum(tree_map[:, k], max_col)

        return is_visible

    def update_visible_map_r(start_range,end_range):
        is_visible = np.zeros(np.shape(tree_map))
        is_visible[-1, :] = 1
        max_row = tree_map[-1,:]
        for k in range(start_range, end_range,-1):
            row_mask = tree_map[k - 1, :] > max_row
            row_mask = [int(x) for x in row_mask]

            # update is_visible
            is_visible[k - 1, :] = [a or b for a, b in zip(row_mask, is_visible[k - 1, :])]

            # update max row
            max_row = np.maximum(tree_map[k-1,:], max_row)

        return is_visible

    def update_visible_map_c_r(start_range, end_range):
        is_visible = np.zeros(np.shape(tree_map))
        is_visible[:, -1] = 1
        max_col = tree_map[:, -1]
        for k in range(start_range, end_range, -1):
            # k = 1
            col_mask = tree_map[:, k-1] > max_col
            col_mask = [int(x) for x in col_mask]

            # update is_visible
            is_visible[:,k-1] = [a or b for a, b in zip(col_mask, is_visible[:, k-1])]

            # update max col
            max_col = np.maximum(tree_map[:,k-1], max_col)
        return is_visible

    # top to bottom
    is_visible_t2b = update_visible_map(1,np.shape(tree_map)[0]-1)
    # # bottom to top
    is_visible_b2t = update_visible_map_r(np.shape(tree_map)[0] - 1, 1)
    # # left to right
    is_visible_l2r = update_visible_map_c(1,np.shape(tree_map)[1]-1)
    # # # # right to left
    is_visible_r2l = update_visible_map_c_r(np.shape(tree_map)[1]-1,1)

    total_visible_map =\
        is_visible_t2b + is_visible_b2t + is_visible_l2r + is_visible_r2l

    total_visible_map = total_visible_map > 0
    total_visible_map = total_visible_map.astype(int)

    print(f'answer part one: {np.sum(total_visible_map)}')

    print(tree_map)

    def check_view_down(start_range, end_range):
        view_score = np.ones(np.shape(tree_map))
        for k in range(start_range, end_range):
            row = np.copy(tree_map[k,:])
            row[0] = 0
            row[-1] = 0

            valid_spots = np.ones(np.shape(row))
            for row_comp_idx in range(k+1,end_range+1):
                if np.all(valid_spots==0):
                    break
                row_comp = tree_map[row_comp_idx,:]
                
                valid_spots[np.where(np.logical_and(row <= row_comp, valid_spots))] = 0
                if not (row_comp_idx + 1) == np.shape(tree_map)[0]:
                    view_score[k,:] += valid_spots


        return view_score

    def check_view_up(start_range, end_range):
        # iterate bottom to top
        view_score = np.ones(np.shape(tree_map))
        for k in reversed(range(start_range, end_range)):
            row = np.copy(tree_map[k, :])
            row[0] = 0
            row[-1] = 0

            valid_spots = np.ones(np.shape(row))
            for row_comp_idx in reversed(range(start_range - 1, k)):
                if np.all(valid_spots == 0):
                    break
                row_comp = tree_map[row_comp_idx, :]

                valid_spots[np.where(np.logical_and(row <= row_comp, valid_spots))] = 0
                if not row_comp_idx == 0:
                    view_score[k, :] += valid_spots

        return view_score

    def check_view_right(start_range, end_range):
        view_score = np.ones(np.shape(tree_map))
        for k in range(start_range, end_range):
            col = np.copy(tree_map[:, k])
            col[0] = 0
            col[-1] = 0

            valid_spots = np.ones(np.shape(col))
            for col_comp_idx in range(k + 1, end_range + 1):
                if np.all(valid_spots == 0):
                    break
                col_comp = tree_map[:,col_comp_idx]

                valid_spots[np.where(np.logical_and(col <= col_comp, valid_spots))] = 0
                if not (col_comp_idx + 1) == np.shape(tree_map)[1]:
                    view_score[:,k] += valid_spots

        return view_score

    def check_view_left(start_range, end_range):
        view_score = np.ones(np.shape(tree_map))
        for k in reversed(range(start_range, end_range)):
            col = np.copy(tree_map[:, k])
            col[0] = 0
            col[-1] = 0

            valid_spots = np.ones(np.shape(col))
            for col_comp_idx in reversed(range(start_range-1, k)):
                if np.all(valid_spots == 0):
                    break
                col_comp = tree_map[:,col_comp_idx]

                valid_spots[np.where(np.logical_and(col <= col_comp, valid_spots))] = 0
                if not col_comp_idx == 0:
                    view_score[:,k] += valid_spots

        return view_score
    
    view_score_down = check_view_down(1,np.shape(tree_map)[0]-1)
    view_score_up = check_view_up(1,np.shape(tree_map)[0]-1)
    view_score_right = check_view_right(1, np.shape(tree_map)[1] - 1)
    view_score_left = check_view_left(1, np.shape(tree_map)[1] - 1)

    for k in [view_score_up, view_score_down, view_score_right, view_score_left]:
        k[np.where(k==0)]=1

    scenic_view = np.multiply(
        np.multiply(view_score_down,view_score_up),
        np.multiply(view_score_right,view_score_left)
    )
    print(f'part 2 solution: {np.max(scenic_view)}')
    # print()

def day9():
    print('oh hi')

    # fname = 'day9_debug.txt'
    fname = 'day9.txt'

    f = open(fname, "r")

    init_size = 1000
    positions_H = np.zeros([init_size,init_size])
    positions_T = np.copy(positions_H)
    positions_history_T = np.copy(positions_T)

    #     head = 1
    #     tail = 2
    #     overlap = 3

    # initialize positions
    initx = init_size - 1
    inity = 0

    initx = int(init_size/2)
    inity = int(init_size/2)

    positions_H[initx,inity] = 1
    positions_T[initx,inity] = 1
    positions_history_T[initx,inity] = 1
    # print(positions_H)

    for idx, line in enumerate(f.readlines()):
        if idx % 100 ==0:
            print(f'line number: {idx}')
        # print(line)
        direction = line.split(' ')[0]
        distance = int(line.split(' ')[-1])

        for k in range(distance):
            # move H
            idx_H = np.asarray(np.where(positions_H == 1)).T
            idx_H_x = idx_H[0][1]
            idx_H_y = idx_H[0][0]
            
            if direction == 'R':
                idx_H_x += 1
            if direction == 'L':
                idx_H_x -= 1
            if direction == 'D':
                idx_H_y += 1
            if direction == 'U':
                idx_H_y -= 1

            if idx_H_x < 0 or idx_H_y < 0:
                raise ValueError('out of grid')

            positions_H[idx_H] = 0
            positions_H[idx_H_y,idx_H_x] = 1

            # move T
            idx_T = np.asarray(np.where(positions_T == 1)).T
            # T/H overlap, do nothing
            if np.all(idx_T[0] == np.array([idx_H_y,idx_H_x])):
                continue
            idx_T_x = idx_T[0][1]
            idx_T_y = idx_T[0][0]
            # one away, do nothing
            if np.abs(idx_H_x - idx_T_x) <= 1 and np.abs(idx_H_y - idx_T_y) <= 1:
                continue
            # diagonal case - trailing (need to move!)
            # todo: issue here, see debug U 4
            # if direction == 'U':
            #     print('debug')
            if (np.abs(idx_H_x - idx_T_x) == 2 and np.abs(idx_H_y - idx_T_y) == 1) or \
                    (np.abs(idx_H_y - idx_T_y) == 2 and np.abs(idx_H_x - idx_T_x) == 1):
            # if np.abs(idx_H_x - idx_T_x) > 1 and np.abs(idx_H_y - idx_T_y) > 1:
                delta_x = (idx_H_x - idx_T_x)
                delta_x /= np.abs(delta_x)
                delta_y = idx_H_y - idx_T_y
                delta_y /= np.abs(delta_y)

                idx_T_x += delta_x
                idx_T_y += delta_y

            # horizontal case
            elif np.abs(idx_H_x - idx_T_x) > 1:
                delta_x = idx_H_x - idx_T_x
                if delta_x > 0:
                    idx_T_x += 1
                else:
                    idx_T_x -= 1

            # vertical case
            elif np.abs(idx_H_y - idx_T_y) > 1:
                delta_y = idx_H_y - idx_T_y
                if delta_y > 0:
                    idx_T_y += 1
                else:
                    idx_T_y -= 1

            positions_T[idx_T] = 0
            idx_T_x = int(idx_T_x)
            idx_T_y = int(idx_T_y)
            positions_T[idx_T_y, idx_T_x] = 1
            positions_history_T[idx_T_y, idx_T_x] = 1

            # print('intermediate')
            # print('H')
            # print(positions_H)
            # print('T')
            # print(positions_T)

        # print('*'* init_size)
        # print('H')
        # print(positions_H)
        # print('T')
        # print(positions_T)
        # # print('T history')
        # # print(positions_history_T)
        # print('*' * init_size)


    # print('*' * init_size)
    # print('final')
    # print('H')
    # print(positions_H)
    # print('T')
    # print(positions_T)
    # print('T history')
    # print(positions_history_T)
    # print('*' * init_size)

    print(f'part 1: {int(np.sum(positions_history_T))}')


def day10():
    print('hello')
    np.set_printoptions(linewidth=200)
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    text_input = 'day10.txt'

    f = open(text_input, "r")

    cycle = 0
    X = 1

    pattern_history = [[0,1]]

    for line in f.readlines():
        if 'noop' in line:
            cycle += 1
        elif 'addx' in line:
            amount = int(line.split(' ')[-1])
            cycle += 2
            X += amount
        else:
            return ValueError
        pattern_history.append([cycle,X])

    history_arr = np.array(pattern_history)
    model = interp1d(history_arr[:,0],history_arr[:,1],kind='previous')

    xnew = np.arange(np.max(history_arr[:,0])+1).astype(int)
    ynew = model(xnew)
    new_arrary = np.vstack([xnew,ynew]).T
    plt.plot(xnew,ynew)
    plt.plot(history_arr[:,0],history_arr[:,1])
    # plt.show()

    idx_to_check = np.arange(0,240,40)+19
    output = np.sum(ynew[idx_to_check] * (idx_to_check + 1))
    # guess 1: 13780 (too high)
    print(f'part 1 output: {output}')

    ## part 2
    screen_output = np.chararray((240)).astype(str)
    screen_output[:] = '.'

    for t in xnew[:-1]:
        # x register
        x = ynew[t]

        # find sprite position
        row = int(t/40)
        sprite_start = x - 1 + row*40
        sprite_end = x + 1 + row*40

        # cursor postion = t
        if sprite_start <= t <= sprite_end:
            screen_output[t] = '#'

    screen_output = np.reshape(screen_output,(6,40))
    print(screen_output)

def day11():
    print('hello')
    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    text_input = 'day11_debug.txt'

    f = open(text_input, "r")
    monkey_dict = {}

    for line in f.readlines():
        if 'Monkey' in line:
            monkey_id = int(line.replace(':\n','').split(' ')[-1])
            monkey_dict[monkey_id] = {}
            continue
        if 'Starting' in line:
            items_str = line.split(':')[-1].split(',')
            items = [int(x) for x in items_str]
            monkey_dict[monkey_id]['items']=items
            continue
        if 'Operation' in line:
            operation_str = line.replace('\n','').split(':')[-1].replace('new =','')
            monkey_dict[monkey_id]['operation'] = operation_str
            continue
        if 'Test' in line:
            monkey_dict[monkey_id]['test'] = int(line.split(' ')[-1])
            continue
        if 'true' in line:
            monkey_dict[monkey_id]['true_monkey'] = int(line.split(' ')[-1])
            continue
        if 'false' in line:
            monkey_dict[monkey_id]['false_monkey'] = int(line.split(' ')[-1])
            continue

    for monkey in monkey_dict:
        monkey_dict[monkey]['inspection_count']= 0
        monkey_dict[monkey]['items_caught'] = []
        
    print('initial monkay dict:')
    pp.pprint(monkey_dict)
    # print(monkey_dict)

    monkey_count = np.max(list(monkey_dict.keys()))+1

    def calc_worry_level(worry0, operation):
        operation_mod = operation.replace('old',str(worry0))
        return eval(operation_mod)

    rounds = 20
    for round in range(rounds):
        print(f'round: {round + 1}')
        for monkey in range(monkey_count):
            items_arr = monkey_dict[monkey]['items']
            
            for idx, item in enumerate(monkey_dict[monkey]['items']):
                update_worry = calc_worry_level(item, monkey_dict[monkey]['operation'])
                update_worry = int(update_worry/3)
                # update_worry = int(update_worry / 1)
                if update_worry % monkey_dict[monkey]['test'] == 0:
                    next_monkey = monkey_dict[monkey]['true_monkey']
                else:
                    next_monkey = monkey_dict[monkey]['false_monkey']
                monkey_dict[next_monkey]['items'].append(update_worry)
                monkey_dict[monkey]['inspection_count'] += 1
            monkey_dict[monkey]['items'] = monkey_dict[monkey]['items'][idx+1:]

    # print(monkey_dict)
    pp.pprint(monkey_dict)

    inspection_count = []

    for monkey in range(monkey_count):
        inspection_count.append(monkey_dict[monkey]['inspection_count'])

    inspection_count = np.array(inspection_count)
    inspection_count = np.sort(inspection_count)
    print('part 1: solution:')
    print(np.multiply(inspection_count[-2],inspection_count[-1]))


def day12():
    # input = 'day12_debug.txt'
    input = 'day12.txt'
    topo_map_str = np.genfromtxt(input,delimiter=1,dtype=str)
    topo_map = np.nan * np.ones(np.shape(topo_map_str))

    # trying Dijkstra's algorithm

    distances = np.inf * np.ones(np.shape(topo_map))
    visited = np.zeros(np.shape(topo_map))

    for iy, ix in np.ndindex(topo_map_str.shape):
        topo_map[iy,ix] = ord(topo_map_str[iy,ix]) - 96

    # starting_point = np.where(topo_map_str=='S')
    # ending_point = np.where(topo_map_str == 'E')

    ending_point = np.where(topo_map_str=='S')
    starting_point = np.where(topo_map_str == 'E')

    distances[starting_point] = 0
    visited[starting_point] = 1

    topo_map[starting_point] = 27
    topo_map[ending_point] = 0

    sx = starting_point[1][0]
    sy = starting_point[0][0]

    ex = ending_point[1]
    ey = ending_point[0]

    cx = sx
    cy = sy

    # def get_valid_neighbors(x, y, topo_map, visited):
    #     # find all neighbors
    #     # neighbors = [
    #     #     [x - 1, y],
    #     #     [x + 1, y],
    #     #     [x, y - 1],
    #     #     [x, y + 1]
    #     # ]
    #     neighbors = [
    #         [y, x - 1],
    #         [y, x + 1],
    #         [y - 1, x],
    #         [y + 1, x]
    #     ]
    #
    #     neighbors = np.array(neighbors)
    #     neighbors = np.reshape(neighbors, (-1, 2))
    #
    #     # filter out neighbors with negative values
    #     neighbors = neighbors[neighbors[:, 0] >= 0]
    #     neighbors = neighbors[neighbors[:, 1] >= 0]
    #
    #     # filter out neighbors that exceed map size
    #     neighbors = neighbors[neighbors[:, 0] < np.shape(topo_map)[0]]
    #     neighbors = neighbors[neighbors[:, 1] < np.shape(topo_map)[1]]
    #
    #     # filter neighbors that have been visited
    #     neighbors = neighbors[visited[neighbors[:, 0], neighbors[:, 1]] == 0]
    #
    #     # filter out neighbors with jumps > 1
    #     current_height = topo_map[y, x]
    #     current_delta_neighbors = topo_map[neighbors[:, 0], neighbors[:, 1]] - current_height
    #     neighbors = neighbors[current_delta_neighbors <= 1]
    #
    #     return neighbors

    def get_valid_neighbors(x, y, topo_map, visited):
        # find all neighbors
        # neighbors = [
        #     [x - 1, y],
        #     [x + 1, y],
        #     [x, y - 1],
        #     [x, y + 1]
        # ]
        neighbors = [
            [y, x - 1],
            [y, x + 1],
            [y - 1, x],
            [y + 1, x]
        ]

        neighbors = np.array(neighbors)
        neighbors = np.reshape(neighbors, (-1, 2))

        # filter out neighbors with negative values
        neighbors = neighbors[neighbors[:, 0] >= 0]
        neighbors = neighbors[neighbors[:, 1] >= 0]

        # filter out neighbors that exceed map size
        neighbors = neighbors[neighbors[:, 0] < np.shape(topo_map)[0]]
        neighbors = neighbors[neighbors[:, 1] < np.shape(topo_map)[1]]

        # filter neighbors that have been visited
        neighbors = neighbors[visited[neighbors[:, 0], neighbors[:, 1]] == 0]

        # filter out neighbors with jumps > 1
        current_height = topo_map[y, x]
        current_delta_neighbors = topo_map[neighbors[:, 0], neighbors[:, 1]] - current_height
        neighbors = neighbors[current_delta_neighbors >= -1]

        return neighbors

    cycles = 0
    while not np.all(visited):
        cycles+=1
        if cycles == 15:
            print('debug')
        print(f'cx: {cx}, cy: {cy}')
        if cx == 0 and cy ==0:
            print('debug')

        # print(visited)
        neighbors = get_valid_neighbors(cx,cy,topo_map, visited)

        # if neighbors = 0, means the algorithm got stuck. Jump back to minimum distance that hasn't been visited
        if len(neighbors) == 0:
            restart_arr = np.copy(distances)
            restart_arr[np.where(visited == 1)] = np.inf
            if np.all(np.isinf(restart_arr)):
                break
            restart_ind = np.where(restart_arr == np.min(restart_arr))
            cy = restart_ind[0][0]
            cx = restart_ind[1][0]
            visited[cy,cx] = 1
            continue
        # print(neighbors)

        # min_distance = np.inf
        neighbor_distances = []

        for neighbor in neighbors:

            neighbor = tuple(neighbor)
            new_distance = distances[cy,cx] + 1
            if np.isinf(new_distance):
                print('debug')
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                

            neighbor_distances.append(distances[neighbor])

        min_path_idx = np.argmin(neighbor_distances)
        next_point = tuple(neighbors[min_path_idx,:])
        print(f'next point: {next_point}')
        visited[next_point] = 1

        cx = next_point[1]
        cy = next_point[0]
        
        # if distances[ending_point][0] < np.inf:
        #     break



    # print(topo_map_str)
    print(topo_map)
    print(distances)
    print(visited)

    number_of_moves = int(distances[ending_point][0])
    print(f'part 1: number of moves: {number_of_moves}')
    # guessed 426 and 427 => too high :(, 400 is too low, 425 was my correct answer

    # part 2: find all distances to point 'a's
    a_distances = distances[np.where(topo_map <= 1)]
    print(a_distances)
    print(f'part 2: min a distance: {np.min(a_distances)}')
    # 420 was too high

    # print(get_valid_neighbors(2,2,topo_map))
            
def day13():
    print('hi')
    input = 'day13_debug.txt'
    with open(input) as f:
        lines = f.read().splitlines()

    left_sides = []
    right_sides = []

    for k in range(0,len(lines),3):
        left_side = eval(lines[k])
        right_side = eval(lines[k+1])
        left_sides.append(left_side)
        right_sides.append(right_side)

    # def side_compare(ls,rs):
    #     for idx, k in enumerate(ls):
    #         if k == len(rs):


    order_results = []
    for idx, (ls, rs) in enumerate(zip(left_sides,right_sides)):
        # if idx > 3:
        #     break
        print(idx)
        compare_sides = np.all(side_compare(ls,rs)) # confirm that all are true
        order_results.append(compare_sides)
        


    # def side_compare_list_of_ints(ls, rs, top=False):
    #     results = []
    #
    #     for k in range(len(ls)):
    #         if k >= len(rs):
    #             results = results + [not top]
    #             break
    #         lse = ls[k]
    #         rse = rs[k]
    #         results = results + side_compare_root(lse,rse)
    #
    #     return results
    #
    # def side_compare_root(ls,rs):
    #     # store results
    #     results = []
    #
    #     if isinstance(ls, int) and isinstance(rs, int):
    #         if ls <= rs:
    #             return [True]
    #         else:
    #             return [False]
    #
    #     # if comparing list vs int, convert int to list
    #     elif isinstance(ls, list) and isinstance(rs, int):
    #         results = results + side_compare_root(ls,[rs])
    #         return results
    #     elif isinstance(ls, int) and isinstance(rs, list):
    #         results = results + side_compare_root([ls],rs)
    #         return results
    #
    #     # both lists
    #     ls_all_ints = np.all([isinstance(x, int) for x in ls])
    #     rs_all_ints = np.all([isinstance(x, int) for x in rs])
    #
    #     if ls_all_ints and rs_all_ints:
    #         results = results + side_compare_list_of_ints(ls, rs)
    #         return results
    #
    #     if len(ls) == 0:
    #         return [True]
    #     if len(rs) == 0:
    #         return [False]
    #
    #     for k in range(len(ls)):
    #         if k > len(rs):
    #             return [True]
    #
    #         results = results + side_compare_root(ls[k], rs[k])
    #
    #     return results
    #
    #
    #     raise NotImplementedError
    #
    # def side_compare(ls,rs):
    #     # store results
    #     results = []
    #
    #     if isinstance(ls, int) and isinstance(rs, int):
    #         return side_compare_root(ls,rs)
    #
    #     # if comparing list vs int, convert int to list
    #     elif isinstance(ls, list) and isinstance(rs, int):
    #         results = results + side_compare_root(ls,[rs])
    #         return results
    #     elif isinstance(ls, int) and isinstance(rs, list):
    #         results = results + side_compare_root([ls],rs)
    #         return results
    #
    #     while(len(ls)>0):
    #         # check if all elements in both lists are ints
    #         ls_all_ints = np.all([isinstance(x, int) for x in ls])
    #         rs_all_ints = np.all([isinstance(x, int) for x in rs])
    #
    #         if ls_all_ints and rs_all_ints:
    #             results = results + side_compare_list_of_ints(ls, rs,top=True)
    #             return results
    #
    #         if len(rs) == 0:
    #             # right side ran out of items
    #             results = results + [False]
    #             break
    #
    #         lse = ls.pop(0)
    #         rse = rs.pop(0)
    #
    #         results = results + side_compare_root(lse, rse)
    #
    #     return results
    #
    # order_results = []
    #
    # # debug 4th element
    # for idx, (ls, rs) in enumerate(zip(left_sides,right_sides)):
    #     # if idx > 3:
    #     #     break
    #     print(idx)
    #     compare_sides = np.all(side_compare(ls,rs)) # confirm that all are true
    #     order_results.append(compare_sides)
    #     # break
    #
    # order_results = np.array(order_results)
    print(order_results)
    expected_results = [True, True, False, True, False, True, False, False]
    print(expected_results)

    # print(f'part 1 answer: {np.sum(np.array(np.where(order_results==True))[0]+1)}')


def day14():
    np.set_printoptions(linewidth=200)
    print('hi')
    # input = 'day14_debug.txt'
    input = 'day14.txt'
    f = open(input, "r")

    points = []

    row_limits = [int(1e6), int(-1e6)]
    col_limits = [int(1e6), int(-1e6)]
    max_row = -np.inf

    for line in f.readlines():
        point_list = []
        for point in line.split('->'):
            point_col = int(point.split(',')[0])
            point_row = int(point.split(',')[1])
            if point_row < row_limits[0]:
                row_limits[0] = point_row
            if point_row > row_limits[1]:
                row_limits[1] = point_row
            if point_col < col_limits[0]:
                col_limits[0] = point_col
            if point_col > col_limits[1]:
                col_limits[1] = point_col

            point_list.append([point_row,point_col])
        points.append(point_list)

    # expand row and col limits
    margin = 200
    row_limits[1] += margin
    col_limits[0] -= margin
    col_limits[1] += margin

    row_limits[0] = 0
    
    print(row_limits)
    print(col_limits)

    # empty space = 0
    # rock structures = 1
    # sand = 2

    cave_map = np.zeros([row_limits[1],col_limits[1]])
    for line_idx in points:
        print(line_idx)
        for k in range(len(line_idx)-1):
            point0 = line_idx[k]
            point1 = line_idx[k+1]
            pt_array = np.array([point0,point1])

            print(f'point0: {point0}, point1: {point1}')

            # draw vertical line
            if len(np.unique(pt_array[:,0]))==1:
                cave_map[pt_array[0,0],np.min(pt_array[:,1]):np.max(pt_array[:,1])+1] = 1
            # draw horizontal line
            elif len(np.unique(pt_array[:,1]))==1:
                cave_map[np.min(pt_array[:,0]):np.max(pt_array[:,0])+1,pt_array[0,1]] = 1
            else:
                raise ValueError('should not get here')

    # cave_map_part2 = np.copy(cave_map)

    # initial map
    print(cave_map[row_limits[0]:row_limits[1],col_limits[0]:col_limits[1]])
    print('start sand!')

    sand_entry_point = [0,500]
    sands = 0 # sand counter
    abyss_hit = False
    while(not abyss_hit):
        moving = True
        # prolly make this a function
        potential_column = sand_entry_point[1]
        potential_row = sand_entry_point[0]

        while(moving):
            if potential_row == np.shape(cave_map)[0] - 1:
                abyss_hit = True
                sands -= 1
                break
            # try to go straight down
            if cave_map[potential_row+1, potential_column] == 0:
                potential_row += 1
            # check for opening on diagonal bottom left
            elif cave_map[potential_row+1, potential_column-1] == 0:
                potential_column = potential_column-1
                potential_row = potential_row + 1
            # check for opening for diagonal bottom right
            elif cave_map[potential_row+1, potential_column+1] == 0:
                potential_column = potential_column+1
                potential_row = potential_row + 1
            else: # both paths are blocked, use the current column and row
                moving = False

        cave_map[potential_row,potential_column] = 2
        sands += 1

    print('cave map with sand!')
    print(cave_map[row_limits[0]:row_limits[1],col_limits[0]:col_limits[1]])

    print(f'part 1: sands until abyss: {sands}')

    cave_map[row_limits[1] - margin + 2,:] = 1
    cave_map[-1,:] = 0

    print(f'part 2: cave map init')
    print(cave_map[row_limits[0]:row_limits[1],col_limits[0]:col_limits[1]])

    print('re-start sand!')

    entry_clogged = False
    while(not entry_clogged):
        if cave_map[sand_entry_point[0],sand_entry_point[1]] == 2:
            entry_clogged = True
            break
        moving = True
        # prolly make this a function
        potential_column = sand_entry_point[1]
        potential_row = sand_entry_point[0]

        while(moving):
            if potential_row == np.shape(cave_map)[0] - 1:
                abyss_hit = True
                sands -= 1
                break
            # try to go straight down
            if cave_map[potential_row+1, potential_column] == 0:
                potential_row += 1
            # check for opening on diagonal bottom left
            elif cave_map[potential_row+1, potential_column-1] == 0:
                potential_column = potential_column-1
                potential_row = potential_row + 1
            # check for opening for diagonal bottom right
            elif cave_map[potential_row+1, potential_column+1] == 0:
                potential_column = potential_column+1
                potential_row = potential_row + 1
            else: # both paths are blocked, use the current column and row
                moving = False

        cave_map[potential_row,potential_column] = 2
        sands += 1

    print(f'part 2: sands till clogged: {sands}')
    print(cave_map[row_limits[0]:row_limits[1], col_limits[0]:col_limits[1]])


def day15():
    print('day 15!')

    # build map

    np.set_printoptions(linewidth=200)
    input_txt = 'day15_debug.txt'
    # input_txt = 'day15.txt'
    f = open(input_txt, "r")

    sensors = []
    beacons = []

    for line in f.readlines():
        sensor_x = int(line.split(' ')[2].replace('x=', '').replace(',', ''))
        sensor_y = int(line.split(' ')[3].replace('y=', '').replace(':', ''))
        beacon_x = int(line.split(' ')[8].replace('x=', '').replace(',', ''))
        beacon_y = int(line.split(' ')[9].replace('y=', '').replace(',', ''))

        sensors.append([sensor_x,sensor_y])
        beacons.append([beacon_x,beacon_y])

    sensors = np.array(sensors)
    beacons = np.array(beacons)

    # grid dimensions
    combined = np.vstack([sensors,beacons])
    min_x = np.min(combined[:, 0])
    max_x = np.max(combined[:, 0])
    min_y = np.min(combined[:, 1])
    max_y = np.max(combined[:, 1])

    print(min_x)
    print(max_x)
    print(sensors)
    print(beacons)

    # margin = 20 # extra space around map
    # tunnel_map = np.zeros([max_y-min_y+margin,max_x-min_x+margin]) # will set minimums to zero
    tunnel_map = np.zeros([max_y - min_y + 1, max_x - min_x + 1])
    # shift these so min values are at 0
    sensors[:, 0] -= min_x
    sensors[:, 1] -= min_y

    beacons[:, 0] -= min_x
    beacons[:, 1] -= min_y

    def find_distance(s,b):
        # find manhatten distance
        dx = int(np.abs(s[0] - b[0]))
        dy = int(np.abs(s[1] - b[1]))

        return dx + dy

    def update_keepout(s,distance,tunnel_map):
        print(f's: {s}')
        print(f'distance: {distance}')
        for k in range(-distance,distance+1):
            y = s[1] + k
            if y < 0 or y > np.shape(tunnel_map)[0] - 1:
                # raise ValueError('map too small')
                continue
            if np.abs(k) == distance:
                x = s[0]
                tunnel_map[y,x] = 3
            else:
                xmin = np.max([s[0] - np.abs(distance-np.abs(k)),0])
                xmax = np.min([s[0] + np.abs(distance-np.abs(k)) + 1,np.shape(tunnel_map)[1]])
                slice_tunnel_map = tunnel_map[y,xmin:xmax]
                slice_tunnel_map[np.where(slice_tunnel_map==0)]=3
                tunnel_map[y,xmin:xmax] = slice_tunnel_map
                tunnel_map[s[1],s[0]]=1

        return tunnel_map

            
    for s, b in zip(sensors, beacons):
        print(s)
        # sensors = 1
        tunnel_map[s[1],s[0]] = 1
        # beacons = 2
        tunnel_map[b[1],b[0]] = 2

    for s, b in zip(sensors, beacons):
        distance = find_distance(s, b)
        tunnel_map = update_keepout(s, distance, tunnel_map)

    row = 10 - min_y + 1

    slice_tunnel_map = tunnel_map[row]

    count = np.sum(slice_tunnel_map>0)
    # # filter out extra columns
    # empty_cols_L = 0
    # empty_cols_R = -1
    #
    # edge_found = False
    # while(not edge_found):
    #     if np.all(tunnel_map[:,empty_cols_L]==0):
    #         empty_cols_L += 1
    #     else:
    #         edge_found = True
    #
    # edge_found = False
    # while(not edge_found):
    #     if np.all(tunnel_map[:,empty_cols_R]==0):
    #         empty_cols_R -= 1
    #     else:
    #         edge_found = True
    #
    # total_empty = empty_cols_L + np.abs(empty_cols_R) - 1

    # part 1 answer
    print(f'part 1 answer: {count}')





    print()

    # s=sensors[6]
    # b=beacons[6]
    # print(s)
    # distance = find_distance(s,b)
    #
    # # keepout = 3
    # tunnel_map = update_keepout(s,distance,tunnel_map)

    print(tunnel_map)

def day16():
    print('hi')

    '''
    define a state as a tuple (current_position, time_left, available_valves), where a 
    valve is only available if I have not opened it. There are 50 positions, 31 different 
    values for time_left and 2^15 possible values for avaialable_valves. Thus the state space 
    for me is between 50 and 51 million. The number of transitions per state is roughly 5 since 
    there aren't a lot of edges in the graph. Therefore it's not too bad, especially 
    considering that 51 million is an upper bound for the number of states, and many of them 
    probably won't be visited during the search
    '''

    import pandas as pd
    from collections import namedtuple

    np.set_printoptions(linewidth=200)
    input_txt = 'day16_debug.txt'
    # input_txt = 'day16.txt'
    f = open(input_txt, "r")

    names = []
    flows = []
    tunnels = []

    for line in f.readlines():
        name = line.split(' ')[1]
        flow = int(line.split('=')[-1].split(';')[0])
        tunnels_ = line.replace(',','').replace('\n','').split(' ')[9:]
        tunnels_.sort()

        names.append(name)
        flows.append(flow)
        tunnels.append(tunnels_)

    print(names)
    print(flows)
    print(tunnels)

    class ValvePath:
        def __init__(self,starting_location,on_valves,time):
            self.map_df = map_df
            self.current_location = starting_location
            self.on_valves = on_valves
            self.time = time
        def next_steps(self):
            sub_map_df = self.map_df[self.map_df.names == self.current_location]
            if sub_map_df.flows == 0:
                # traverse to new tunnel
                next_tunnels = sub_map_df.tunnels

    map_df = pd.DataFrame({'names': names,
                           'flows': flows,
                           'tunnels': tunnels
                           })

    TState = namedtuple('TState', 'current_tunnel closed_valves time_left release_rate released')
    open_valves = list(map_df[map_df['flows']>0].names)

    
    print(map_df)
    print(map_df.sort_values('flows', ascending=False))

    starting_point = 'AA'
    all_paths = []

    init_state = TState(starting_point,open_valves,30,0,0)

    def next_state_calc(state, map_df):
        # update total released
        released = state.released + state.release_rate

        next_states = []
        if state.current_tunnel in state.closed_valves:
            # open valve where we are
            release_rate = state.release_rate + map_df[map_df.names == state.current_tunnel].iloc[0].flows

            closed_valves = state.closed_valves.copy()
            closed_valves.remove(state.starting_point)

            next_states.append(TState(
                state.current_tunnel,
                closed_valves,
                state.time_left - 1,
                release_rate,
                released
            ))

        next_tunnels = map_df[map_df.names == state.current_tunnel].iloc[0].tunnels

        for k in next_tunnels:
            next_states.append(
                TState(
                    k,
                    state.closed_valves,
                    state.time_left - 1,
                    state.release_rate,
                    released
                )
            )

        return next_states

    print(init_state)
    print(next_state_calc(init_state,map_df))


    # find all next steps





    # map_df.sort_values('flows', ascending=False, inplace=True)

    # def travel_time(start,destination,map_df):
    #
    #
    #
    # # start with highest flow, find time to get to starting point from there
    # time_to_travel = []
    # current_tunnel = 'AA'
    # for index, row in map_df.iterrows():
    #     time_to_travel.apppend(travel_time(row['names'],current_tunnel))



    # print()


def day17():
    print('hi')
    from collections import namedtuple

    fname = 'day17_debug.txt'
    f = open(fname, "r")

    for line in f.readlines():
        directions = list(line.replace('\n',''))


    # define shapes
    RShape = namedtuple('RShape', 'lowest_point_indices highest_point_indices')

    hline_shape = RShape((0,0,0,0),(1,1,1,1))
    plus_shape = RShape((1,0,1),(1,2,1))
    corner_shape = RShape((0, 0, 0), (1, 1, 2))
    vline_shape = RShape((0),(3))
    square_shape = RShape((0,0),(1,1))

    shapes = [hline_shape, plus_shape, corner_shape, vline_shape, square_shape]

    # initialize all heights to 0
    rock_heights = np.zeros((1, 7))


    cycles = 1

    for k in range(cycles):
        # place rock in plot
        highest_point = np.max(rock_heights)

def day18():
    fname = 'day18_debug.txt'
    # fname = 'day18.txt'
    from numpy import genfromtxt
    import matplotlib.pyplot as plt
    raw_cubes = genfromtxt(fname, delimiter=',',dtype=int)

    part1 = True
    if part1:
        processed_cubes = []

        for k in range(np.shape(raw_cubes)[0]):
            if k % 100 == 0:
                print(f'cycle: {k}')
            raw_cube = np.hstack([raw_cubes[k,:],6])
            # print('raw cube')
            # print(raw_cube)
            for pc in processed_cubes:
                # check if two points are common
                if (pc[0] == raw_cube[0]) & (pc[1] == raw_cube[1]) & (np.abs(pc[2] - raw_cube[2]) <= 1) or \
                        (pc[0] == raw_cube[0]) & (pc[2] == raw_cube[2]) & (np.abs(pc[1] - raw_cube[1]) <= 1) or \
                        (pc[1] == raw_cube[1]) & (pc[2] == raw_cube[2]) & (np.abs(pc[0] - raw_cube[0]) <= 1):
                    pc[3] -= 1
                    raw_cube[3] -= 1
            processed_cubes.append(raw_cube)
            # print(processed_cubes)

        # for k in range(np.shape(processed_cubes)[0]):
        #     pc = processed_cubes[k]
        #     for k in


        processed_cubes = np.array(processed_cubes)
        print(processed_cubes)
        print(f'part 1 answer: {np.sum(processed_cubes[:,3])}')


    # # print(my_data)
    #
    # raw_cubes = raw_cubes - 1
    #
    # volume_space = np.zeros(np.max(raw_cubes,axis=0)+1)
    #
    # for k in range(np.shape(raw_cubes)[0]):
    #     volume_space[tuple(raw_cubes[k,:])] = 1
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(volume_space[:,0],volume_space[:,1],volume_space[:,2])
    # plt.show()


    print(raw_cubes)
    # create numpy array to contain all the space

def day19():
    print('hi 19')
    from copy import deepcopy

    fname = 'day19_debug.txt'
    f = open(fname, "r")
    
    blueprint_number = []
    ore_robot_cost = []
    clay_robot_cost = []
    obsidian_robot_cost = []
    geode_robot_cost = []

    for line in f.readlines():
        bp = int(line.split(' ')[1].replace(':',''))
        ore = int(line.split(' ')[6])
        clay = int(line.split(' ')[12])
        obsidian = (int(line.split(' ')[18]),int(line.split(' ')[21]))
        geode = (int(line.split(' ')[27]),int(line.split(' ')[30]))

        blueprint_number.append(bp)
        ore_robot_cost.append(ore)
        clay_robot_cost.append(clay)
        obsidian_robot_cost.append(obsidian)
        geode_robot_cost.append(geode)

    '''
    initialize state zero
    # tuple decoder
    index 0 = ore robots
    index 1 = clay robots
    index 2 = obsidian robots
    index 3 = geode robots
    index 4 = ore count
    index 5 = clay count
    index 6 = obsidian count
    index 7 = geode count
    index 8 = time remaining
    index 9 = open geode count 
    '''

    initial_state = np.zeros([10,],dtype=int)
    initial_state[0] = 1
    initial_state[8] = 24

    idx = 0 # blueprint index
    all_states = [initial_state]

    def next_state(current_state,bp_idx):
        next_states = []

        # update resources for end of time period
        resources = np.add(current_state[4:8],current_state[0:4])

        # don't do anything this step - just collect resources
        next_state = np.copy(current_state)
        next_state[4:8] = resources
        # next_state = np.concatenate((current_state[0:3],resources))
        next_state[8] -= 1
        next_states.append(next_state)

        # check if enough original resources to build ore robots
        if current_state[4] >= ore_robot_cost[idx]:
            next_state = np.copy(current_state)
            next_state[4:8] = resources
            next_state[4] -= ore_robot_cost[idx]
            next_state[0] += 1
            next_state[8] -= 1 # maybe move to later (common step)
            next_states.append(next_state)

        # check if enough original resources to build clay robots
        if current_state[4] >= clay_robot_cost[idx]:
            next_state = np.copy(current_state)
            next_state[4:8] = resources
            next_state[4] -= clay_robot_cost[idx]
            next_state[1] += 1
            next_state[8] -= 1 # maybe move to later (common step)
            next_states.append(next_state)

        # check if enough original resources to build obsidian robots
        if (current_state[4] >= obsidian_robot_cost[idx][0]) and (current_state[5] >= obsidian_robot_cost[idx][1]):
            next_state = np.copy(current_state)
            next_state[4:8] = resources
            next_state[4] -= obsidian_robot_cost[idx][0]
            next_state[5] -= obsidian_robot_cost[idx][1]
            next_state[2] += 1
            next_state[8] -= 1  # maybe move to later (common step)
            next_states.append(next_state)

        # check if enough original resources to build geode robots
        if (current_state[4] >= geode_robot_cost[idx][0]) and (current_state[6] >= geode_robot_cost[idx][1]):
            next_state = np.copy(current_state)
            next_state[4:8] = resources
            next_state[4] -= geode_robot_cost[idx][0]
            next_state[6] -= geode_robot_cost[idx][1]
            next_state[3] += 1
            next_state[8] -= 1  # maybe move to later (common step)
            next_states.append(next_state)

        return next_states

    for k in range(12):
        print(all_states)
        cycle_states = []
        for state in all_states:
            if state[8] > initial_state[8] - k:
                continue # don't reprocess past timesteps
            next_states = next_state(state,idx)
            cycle_states = cycle_states + next_states
        all_states = all_states + cycle_states

    print(all_states)





    #
    # for k in range(2):
    #     # check if enough resources to build robots
    #
    #
    #     # update resource counts
    #     total_resources = np.add(total_resources,robot_count)
    #
    #     print('total resources')
    #     print(total_resources)
    #     print('robot count')
    #     # print(robot_count)

    # print(blueprint_number)
    # print(ore_robot_cost)
    # print(clay_robot_cost)
    # print(obsidian_robot_cost)
    # print(geode_robot_cost)

def day20():
    print('hi')

    fname = 'day20_debug.txt'
    fname = 'day20.txt'
    orig_message = np.genfromtxt(fname,dtype=int)

    message = np.vstack([orig_message,np.zeros(len(orig_message),dtype=int)]).T
    # add column of 0s to track which ones have been modified that round
    print(message)

    rounds = 1
    for k in range(rounds):
        debug_counter=0
        while(np.any(message[:,1]==0)):
            # if debug_counter>2:
            #     break
            debug_counter+=1
            # find first unmodified index
            min_idx = np.argmin(message[:,1])
            element = np.copy(message[min_idx,:])
            element[1] = 1 # update to indicate that it has moved
            message = np.delete(message,min_idx,axis=0)
            updated_index = min_idx + element[0]
            if np.abs(updated_index) > np.shape(message)[0]:
                updated_index = np.abs(updated_index) % np.shape(message)[0]
                updated_index *= -1
                updated_index += 1

            elif updated_index == 0 and element[0]<0:
                updated_index -= 1
            elif updated_index > np.shape(message)[0]:
                updated_index = updated_index % np.shape(message)[0]
                updated_index -= 1
                # print(f'wrap around: updated index {updated_index}')
            message = np.insert(message, updated_index, element, axis=0)



    message = message[:,0]
    print(message)
    offset = -3
    idx_1000 = ((1000) % len(message)) + offset
    idx_2000 = ((2000) % len(message)) + offset
    idx_3000 = ((3000) % len(message)) + offset
    
    print(idx_1000)
    print(idx_2000)
    print(idx_3000)

    print(f'1000th element: {message[idx_1000]}')
    print(f'2000th element: {message[idx_2000]}')
    print(f'3000th element: {message[idx_3000]}')
          
    print(f'part 1 solution: {message[idx_1000] + message[idx_2000] + message[idx_3000]}')

def day22():
    print('hi')
    np.set_printoptions(linewidth=200)
    np.set_printoptions(suppress=True)

    fname = 'day22_debug.txt'
    fname = 'day22.txt'
    f = open(fname, "r")

    line_length = 0
    lines = []
    for idx, line in enumerate(f.readlines()):
        # print(idx)
        # print(line)
        line_length = np.max([line_length,len(line)-1]) # get longest line length
        line_list = list(line)
        line_list=line_list[:-1]
        line_list = [np.nan if x == ' ' else x for x in line_list]
        line_list = [0 if x == '.' else x for x in line_list]
        line_list = [1 if x == '#' else x for x in line_list]
        line_arr = np.array(line_list)

        lines.append(line_arr)

    # directions on last line
    directions = line
    print(directions)

    rows = idx - 2 + 1

    gps_map = np.nan*np.ones([rows, line_length])
    cube_map = np.copy(gps_map)

    for idx, line_arr in enumerate(lines[:-2]):
        gps_map[idx,:len(line_arr)] = line_arr

    # find faces of cube
    if 'debug' in fname:
        cube_edge = 4
    else:
        cube_edge = 50

    # find first corner of face 1
    face = 1
    idx_row_face = 0
    col_face_tracker = 0
    idx_col_face = np.where(~np.isnan(gps_map[idx_row_face, :]))[0][col_face_tracker]

    while(face<7):
        cube_map[idx_row_face:idx_row_face+cube_edge,idx_col_face:idx_col_face+cube_edge] = face
        face += 1

        if face==7:
            break

        if ((idx_col_face + cube_edge) >= np.shape(gps_map)[1]) or  np.isnan(gps_map[idx_row_face,idx_col_face+cube_edge]):
            col_face_tracker = 0
            idx_row_face += cube_edge # if nan, shift by several rows
            idx_col_face = np.where(~np.isnan(gps_map[idx_row_face, :]))[0][col_face_tracker]
        else:
            # shift columns
            col_face_tracker += cube_edge
            idx_col_face = np.where(~np.isnan(gps_map[idx_row_face, :]))[0][col_face_tracker]
            
    print(cube_map)
    raise NotImplementedError

    current_tile_row = 0
    current_tile_col = np.where(gps_map[current_tile_row,:]==0)[0][0]
    '''
    facing
    Up = 270
    Right = 0
    Down = 90
    Left = 180
    '''

    def move_distance(row, col, distance, facing, gps_map):
        if facing == 0:
            for k in range(distance):
                col += 1

                if col == np.shape(gps_map)[1] or np.isnan(gps_map[row,col]):
                    # wrap around
                    col_free_or_block = np.where(~np.isnan(gps_map[row,:]))[0][0]
                    if gps_map[row,col_free_or_block] == 0:
                        col = col_free_or_block
                    else:
                        col -= 1 # go back to prior space
                elif  gps_map[row,col] == 1:
                    col -= 1 # hit rock

        elif facing == 180:
            for k in range(distance):
                col -= 1
                if col == -1 or np.isnan(gps_map[row,col]):
                    # wrap around
                    col_free_or_block = np.where(~np.isnan(gps_map[row,:]))[0][-1]
                    if gps_map[row,col_free_or_block] == 0:
                        col = col_free_or_block
                    else:
                        col += 1 # go back to prior space
                elif gps_map[row,col] == 1:
                    col += 1 # hit rock

        elif facing == 90:
            for k in range(distance):
                row += 1
                if row == np.shape(gps_map)[0] or np.isnan(gps_map[row,col]):
                    # wrap around
                    row_free_or_block = np.where(~np.isnan(gps_map[:,col]))[0][0]
                    if gps_map[row_free_or_block,col] == 0:
                        row = row_free_or_block
                    else:
                        row -= 1 # go back to prior space
                elif gps_map[row,col] == 1:
                    row -= 1 # hit rock

        elif facing == 270:
            for k in range(distance):
                row -= 1
                if row == -1 or np.isnan(gps_map[row,col]):
                    # wrap around
                    row_free_or_block = np.where(~np.isnan(gps_map[:,col]))[0][-1]
                    if gps_map[row_free_or_block,col] == 0:
                        row = row_free_or_block
                    else:
                        row += 1 # go back to prior space
                elif gps_map[row,col] == 1:
                    row += 1 # hit rock

        return row, col

    current_facing = 0

    # parse directions
    distance = []
    for k in directions:
        if k.isalpha():
            # check to see if any distance measurement in the queue
            if len(distance) > 0:
                distance_int = ''.join(distance)
                distance_int = int(distance_int)
                distance = []
                current_tile_row, current_tile_col = move_distance(current_tile_row,
                                                                   current_tile_col,
                                                                   distance_int,
                                                                   current_facing,
                                                                   gps_map)
        if k == 'R':
            current_facing += 90
            if current_facing == 360:
                current_facing = 0
        elif k =='L':
            current_facing -= 90
            if current_facing == -90:
                current_facing = 270
        elif k.isdigit():
            distance.append(k)

        current_map = np.copy(gps_map)
        current_map[current_tile_row,current_tile_col] = 1000 + current_facing
        # current_map.astype(int)
        # print('\n')
        # print(current_map)

    # check to see if any distance measurement in the queue
    if len(distance) > 0:
        distance_int = ''.join(distance)
        distance_int = int(distance_int)
        distance = []
        current_tile_row, current_tile_col = move_distance(current_tile_row,
                                                           current_tile_col,
                                                           distance_int,
                                                           current_facing,
                                                           gps_map)

    print('\n')
    print(current_map)

    part1_score = 1000 * (current_tile_row + 1) + 4 * (current_tile_col + 1) + current_facing/90

    print(f'part 1 score: {int(part1_score)}')

def day23():
    print('hi')
    np.set_printoptions(linewidth=2000)

    # fname = 'day23_debug.txt'
    fname = 'day23.txt'
    f = open(fname, "r")

    lines = []
    for idx, line in enumerate(f.readlines()):
        line_list = list(line.replace('\n',''))
        line_list = [0 if x == '.' else x for x in line_list]
        line_list = [1 if x == '#' else x for x in line_list]
        line_arr = np.array(line_list)

        lines.append(line_arr)

    margin = 300
    forest_map = np.zeros([len(lines)+int(2*margin), len(lines[0])+int(2*margin)],dtype=int)
    for idx, line_arr in enumerate(lines):
        forest_map[margin+idx,margin:(margin+len(line_arr))] = line_arr

    print(forest_map)

    # moves
    move_order = ['n','s','w','e']
    m_ind = 0
    elves_moving = True
    rounds = 0

    # for k in range(10):
    while(elves_moving):
        rounds+=1
        if rounds % 5 == 0:
            print(f'rounds: {rounds}')
        current_loc = np.where(forest_map==1)
        proposed_move_direction = move_order[m_ind]

        elf_status = np.vstack([current_loc[0],current_loc[1],np.zeros([3,len(current_loc[0])])]).T
        elf_status = elf_status.astype(int)

        '''
        elf_status
        col 0 = orig row
        col 1 = orig col
        col 2 = proposed new location found (0 = no, 1 = yes)
        col 3 = new row
        col 4 = new col
        '''

        while (np.any(elf_status[:, 2]==0)):
            for elf_idx in range(np.shape(elf_status)[0]):
                new_loc_found = elf_status[elf_idx, 2]
                if new_loc_found:
                    continue

                og_row = elf_status[elf_idx,0]
                og_col = elf_status[elf_idx,1]

                # check if clear all around
                if np.all(forest_map[og_row - 1,og_col-1:og_col+2]==0) and \
                    np.all(forest_map[og_row + 1, og_col - 1:og_col + 2] == 0) and \
                    forest_map[og_row, og_col-1] == 0 and \
                    forest_map[og_row, og_col+1] == 0:
                    elf_status[elf_idx, 2:] = np.array([1, og_row, og_col])
                    continue

                for d in move_order:
                    if d == 'n':
                        n_row = og_row - 1
                        if np.all(forest_map[n_row,og_col-1:og_col+2]==0):
                            elf_status[elf_idx,2:] = np.array([1,n_row,og_col])
                            break
                    if d == 's':
                        n_row = og_row + 1
                        if np.all(forest_map[n_row,og_col-1:og_col+2]==0):
                            elf_status[elf_idx,2:] = np.array([1,n_row,og_col])
                            break
                    if d == 'w':
                        n_col = og_col - 1
                        if np.all(forest_map[og_row-1:og_row+2,n_col]==0):
                            elf_status[elf_idx,2:] = np.array([1,og_row,n_col])
                            break
                    if d == 'e':
                        n_col = og_col + 1
                        if np.all(forest_map[og_row-1:og_row+2,n_col]==0):
                            elf_status[elf_idx,2:] = np.array([1,og_row,n_col])
                            break

                # add case where all spots are block
                if elf_status[elf_idx,2] == 0:
                    # print('no spots found')
                    elf_status[elf_idx, 2:] = np.array([1, og_row, og_col])

        if np.all(elf_status[:,0] == elf_status[:,3]) and np.all(elf_status[:,1] == elf_status[:,4]):
            elves_moving = False
            break

        # rounds+=1

        # check if any elves are trying to go to the same spot
        unique_arr, cnts = np.unique(elf_status[:, 3:], axis=0, return_counts=True)

        for elf in range(np.shape(elf_status)[0]):
            if cnts[np.where((unique_arr==np.array(elf_status[elf,3:])).all(axis=1))][0] > 1:
                elf_status[elf,3:] = elf_status[elf,0:2]

        # update move order
        move_order = list(move_order[1:]) + list(move_order[0])
            # np.where(unique_arr==np.array(elf_status[elf,3:]))
            
        # print('\n')
        # print(elf_status)
        forest_map=np.zeros(np.shape(forest_map), dtype=int)
        forest_map[(elf_status[:,3],elf_status[:,4])] = 1
        # print('forest map')
        # print(forest_map)

        #
        # for idx, (r,c) in enumerate(zip(current_loc[0],current_loc[1])):
        #     print(f'r: {r}, c: {c}')
        #     if proposed_move_direction == 'n':
        #         proposed_row = r - 1
        #         if np.all(forest_map[proposed_row,c-1:c+2]==0):
        #             proposed_rows.append(proposed_row)
        #             proposed_cols.append(c)
        #             elf_idx.append(idx)





        # if proposed_move_direction is 'n':
        #     proposed_loc = (current_loc[0],current_loc[1]-1)
        #     proposed_status = forest_map[proposed_loc]

    # crop to min rectangle
    nz = np.nonzero(forest_map)  # Indices of all nonzero elements
    forest_map_trimmed = forest_map[nz[0].min():nz[0].max()+1,
                      nz[1].min():nz[1].max()+1]

    print(forest_map_trimmed)
    empty_count = np.count_nonzero(forest_map_trimmed==0)

    print(f'part 1 answer: {empty_count}')

    print(f'part 2: rounds {rounds}')

def day24():
    print('almost done!')

    fname = 'day24_debug.txt'

    f = open(fname, "r")

    '''
    blizzard directions:
    ^ = 2
    > = 3
    v = 4
    < = 5
    '''

    lines = []
    for idx, line in enumerate(f.readlines()):
        line_list = list(line.replace('\n',''))
        line_list = [0 if x == '.' else x for x in line_list]
        line_list = [np.nan if x == '#' else x for x in line_list]
        line_list = [2 if x == '^' else x for x in line_list]
        line_list = [3 if x == '>' else x for x in line_list]
        line_list = [4 if x == 'v' else x for x in line_list]
        line_list = [5 if x == '<' else x for x in line_list]

        lines.append(line_list)

    map_arr = np.array(lines)
    blizzard_directions_up = np.zeros(np.shape(map_arr))
    blizzard_directions_up[np.where(map_arr==2)] = 1

    blizzard_directions_right = np.zeros(np.shape(map_arr))
    blizzard_directions_right[np.where(map_arr == 3)] = 1

    blizzard_directions_down = np.zeros(np.shape(map_arr))
    blizzard_directions_down[np.where(map_arr==4)] = 1

    blizzard_directions_left = np.zeros(np.shape(map_arr))
    blizzard_directions_left[np.where(map_arr==5)] = 1

    map_arr[np.where(map_arr > 1)] = 0

    blizzard_directions_up += map_arr
    blizzard_directions_right += map_arr
    blizzard_directions_down += map_arr
    blizzard_directions_left += map_arr

    blizzard_directions = np.array([blizzard_directions_up,blizzard_directions_right,blizzard_directions_down,blizzard_directions_left])
    print(np.shape(blizzard_directions))

    print(map_arr)

    # simulate blizzard motion
    minutes = 10

    print(f'blizzards at 0 minutes')
    print(np.sum(blizzard_directions,axis=0))

    for m in range(minutes):
        # up
        dir_idx = 0
        r_updates = []
        c_updates = []

        if np.any(blizzard_directions[dir_idx] > 0):
            for r, c in zip(np.where(blizzard_directions[dir_idx] > 0)[0],
                            np.where(blizzard_directions[dir_idx] > 0)[1]):
                print(f'r: {r},c: {c}')
                rnew = r - 1
                if rnew == -1:
                    rnew = np.shape(blizzard_directions)[dir_idx] - 2

                elif np.isnan(blizzard_directions[dir_idx][rnew][c]):
                    if np.isnan(blizzard_directions[dir_idx][-1][c]):
                        rnew = np.shape(blizzard_directions)[1] - 2
                    else:
                        rnew = np.shape(blizzard_directions)[1] - 2

                r_updates.append(rnew)
                c_updates.append(c)

            blizzard_directions[dir_idx][np.where(blizzard_directions[dir_idx] > 0)] = 0
            for r, c in zip(r_updates, c_updates):
                blizzard_directions[dir_idx][r][c] = 1

        # right
        print('right')
        dir_idx = 1
        r_updates = []
        c_updates = []

        if np.any(blizzard_directions[dir_idx]>0):
            for r, c in zip(np.where(blizzard_directions[dir_idx]>0)[0],
                            np.where(blizzard_directions[dir_idx]>0)[1]):
                print(f'r: {r},c: {c}')
                cnew = c + 1
                if np.isnan(blizzard_directions[dir_idx][r][cnew]):
                    cnew = 1 # wrap to other side

                r_updates.append(r)
                c_updates.append(cnew)

            blizzard_directions[dir_idx][np.where(blizzard_directions[dir_idx]>0)] = 0
            for r, c in zip(r_updates,c_updates):
                blizzard_directions[dir_idx][r][c] = 1

        # down
        dir_idx = 2
        r_updates = []
        c_updates = []

        if np.any(blizzard_directions[dir_idx] > 0):
            for r, c in zip(np.where(blizzard_directions[dir_idx] > 0)[0],
                            np.where(blizzard_directions[dir_idx] > 0)[1]):
                print(f'r: {r},c: {c}')
                rnew = r + 1
                if rnew >= np.shape(blizzard_directions)[1] or np.isnan(blizzard_directions[dir_idx][rnew][c]):
                    if np.isnan(blizzard_directions[dir_idx][0][c]):

                        rnew = 1  # wrap to other side
                    else:
                        rnew = 0

                r_updates.append(rnew)
                c_updates.append(c)

            blizzard_directions[dir_idx][np.where(blizzard_directions[dir_idx] > 0)] = 0
            for r, c in zip(r_updates, c_updates):
                blizzard_directions[dir_idx][r][c] = 1

        # left
        dir_idx = 3
        r_updates = []
        c_updates = []
        
        if np.any(blizzard_directions[dir_idx] > 0):
            for r, c in zip(np.where(blizzard_directions[dir_idx] > 0)[0],
                            np.where(blizzard_directions[dir_idx] > 0)[1]):
                print(f'r: {r},c: {c}')
                cnew = c - 1
                if np.isnan(blizzard_directions[dir_idx][r][cnew]):
                    cnew = np.shape(blizzard_directions)[1]-2  # wrap to other side

                r_updates.append(r)
                c_updates.append(cnew)

            blizzard_directions[dir_idx][np.where(blizzard_directions[dir_idx] > 0)] = 0
            for r, c in zip(r_updates, c_updates):
                blizzard_directions[dir_idx][r][c] = 1

        print(f'blizzards at {m+1} minutes')
        print(np.sum(blizzard_directions,axis=0))
        # print(blizzard_directions[1])
        input()

def day25():
    print('almost done!')

    fname = 'day25_debug.txt'
    fname = 'day25.txt'

    f = open(fname, "r")

    def snafu2decimal_digit(digit):
        if digit == '=':
            return -2
        if digit == '-':
            return -1
        if digit == '0':
            return 0
        if digit == '1':
            return 1
        if digit == '2':
            return 2

    all_values =[]
    for idx, line in enumerate(f.readlines()):
        line = line.replace('\n','')
        
        numbers_place = 0
        line_total = 0
        elements = list(line)
        for k in reversed(elements):
            line_total += snafu2decimal_digit(k) * (5 ** numbers_place)
            numbers_place+=1
        all_values.append(line_total)
        print(line_total)

    cum_sum = np.sum(all_values)
    print(f'cum sum: {cum_sum}')

    '''
    125 25 5 0
    
 SNAFU  Decimal
1=-0-2     1747
 12111      906
  2=0=      198
    21       11
  2=01      201
   111       31
 20012     1257
   112       32
 1=-1=      353
  1-12      107
    12        7
    1=        3
   122       37
   
   Powers of 5
   0    1
   1    5
   2    25
   3    125
   4    625
   5    3125
   
    '''

    def decimal2snafu(number):
        residual = number
        digits = []

        while residual > 0:
            check_digit = residual % 5
            if check_digit == 0:
                digits.append(0)
            elif check_digit == 1:
                digits.append(1)
            elif check_digit == 2:
                digits.append(2)
            elif check_digit == 3:
                digits.append(-2)
            elif check_digit == 4:
                digits.append(-1)

            if check_digit <=2:
                residual = (residual - check_digit)/5
            else:
                residual = (residual - digits[-1]) / 5

        digits = [str(x) for x in digits]
        digits = [x.replace('-2','=').replace('-1','-') for x in digits]
        digits.reverse()
        snafu_num = ''.join(x for x in digits)

        return snafu_num

    print('test')
    print(decimal2snafu(15))
    print(f'1=-0-2     1747:  {decimal2snafu(1747)}')
    print(f'12111     906:  {decimal2snafu(906)}')
    print(f'21     11:  {decimal2snafu(11)}')
    print(f'part 1 answer: {decimal2snafu(cum_sum)}')


    print()

    



        




        
if __name__ == '__main__':
    day = 9
    if day == 1:
        day1()
    elif day == 2:
        day2()
    elif day == 3:
        day3()
    elif day == 4:
        day4()
    elif day == 5:
        day5()
    elif day == 6:
        day6()
    elif day == 7:
        day7()
    elif day == 8:
        day8()
    elif day == 9:
        day9()
    elif day == 10:
        day10()
    elif day == 11:
        day11()
    elif day == 12:
        day12()
    elif day == 13:
        day13()
    elif day == 14:
        day14()
    elif day == 15:
        day15()
    elif day == 16:
        day16()
    elif day == 17:
        day17()
    elif day == 18:
        day18()
    elif day == 19:
        day19()
    elif day == 20:
        day20()
    elif day == 22:
        day22()
    elif day == 23:
        day23()
    elif day == 24:
        day24()
    elif day == 25:
        day25()

    else:
        raise NotImplementedError(f'Day {day} not implemented yet')