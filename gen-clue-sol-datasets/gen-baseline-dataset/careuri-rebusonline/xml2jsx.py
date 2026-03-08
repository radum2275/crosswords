##/usr/bin/python
#coding=utf-8

import sys
import xml.etree.ElementTree as et
import pprint
import argparse
import io
import re

def build_records(grid, horiz, vert):
    '''Build all the records'''
    records = []
    rows = grid.split("\n")
    nr_rows = len(rows)
    nr_cols = len(rows[0])
    horiz_clues = re.split(" - | – | - | - ", horiz.decode())
    horiz_clue_idx = 0
    position_dic = {}
    pos = 1
    for row in range(0, nr_rows):
        slots = rows[row].split(".")
        nr_slots = 0
        startx = 0
        new_row = 1
        for slot in slots:
            if len(slot) > 1:
                new_rec = {}
                nr_slots = nr_slots + 1
                sys.stderr.write("%s -- %s\n" %(slot, horiz_clues[horiz_clue_idx]))
                if new_row == 1:
                    new_rec["clue"] = "%d)\t\t%s" %(row + 1, horiz_clues[horiz_clue_idx])
                    new_row = 0
                else:
                    new_rec["clue"] = "--\t\t%s" %(horiz_clues[horiz_clue_idx])
                horiz_clue_idx = horiz_clue_idx + 1
                new_rec["answer"] = slot
                new_rec["starty"] = row + 1
                new_rec["startx"] = startx + 1
                new_rec["orientation"] = "across"
                key = new_rec["startx"]*nr_rows+new_rec["starty"]
                if key not in position_dic:
                    position_dic[key] = pos
                    pos = pos + 1
                new_rec["position"] = position_dic[key]
                records.append(new_rec)
            startx = startx + len(slot) + 1
    vert_clues = re.split(" - | – | - | - ", vert.decode())
    vert_clue_idx = 0
    for col in range(0, nr_cols):
        stringcol = ""
        for row in range(0, nr_rows):
            stringcol = stringcol + rows[row][col]
        slots = stringcol.split(".")
        nr_slots = 0
        starty = 0
        new_col = 1
        for slot in slots:
            if len(slot) > 1:
                new_rec = {}
                nr_slots = nr_slots + 1
                sys.stderr.write("%s -- %s\n" %(slot, vert_clues[vert_clue_idx]))
                if new_col == 1:
                    new_rec["clue"] = "%d)\t\t%s" %(col + 1, vert_clues[vert_clue_idx])
                    new_col = 0
                else:
                    new_rec["clue"] = "--\t\t%s" %(vert_clues[vert_clue_idx])
               # print (horiz)
                #new_rec["clue"] = u"{}".format(horiz_clues[horiz_clue_idx])
                vert_clue_idx = vert_clue_idx + 1
                new_rec["answer"] = slot
                new_rec["starty"] = starty + 1
                new_rec["startx"] = col + 1
                new_rec["orientation"] = "down"
                key = new_rec["startx"]*nr_rows+new_rec["starty"]
                if key not in position_dic:
                    position_dic[key] = pos
                    pos = pos + 1
                new_rec["position"] = position_dic[key]
                #new_rec["clue"] = "%d)\t\t%s" %(position_dic[key], new_rec["clue"])
                records.append(new_rec)
            starty = starty + len(slot) + 1
    return records

def read_puzzle(filename):
    '''Read the puzzle from the file'''
#    with io.open(filename, "r", encoding="utf-8") as file:
    with io.open(filename, "r") as file:
        tree = et.fromstring(file.read())
    for grid_elem in tree.iterfind('grid'):
        grid = grid_elem.text.strip()
    for title_elem in tree.iterfind('title'):
        title = title_elem.text.strip()
    for info_elem in tree.iterfind('info_to_display'):
        info = info_elem.text.strip()
    for horiz_clues in tree.iterfind('horizontal_clues'):
        horiz = horiz_clues.text.strip().encode("utf-8")
    for vert_clues in tree.iterfind('vertical_clues'):
        vert = vert_clues.text.strip().encode("utf-8")
    info = info.replace("\n", "<br>")
    result = {}
    # This version does not support symbols such as | in the grid
    result["grid"] = grid.replace("|", "")
    result["title"] = title
    result["horiz_clues"] = horiz
    result["vert_clues"] = vert
    result["info"] = info
    return result


def write_as_jsx(recs, title, info):
    print("// File structure based on [c] Jesse Weisbeck, MIT/GPL")
    print("(function($) {")
    print("  $(function() {")
    print("    var puzzleData = ")
    pprint.pprint(RECS, indent=6, width=500)
    print("    $('#puzzle-wrapper').crossword(puzzleData, `%s`, '%s');" %(title, info))
    print("  })")
    print("})(jQuery)")

def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filename', type=str,
                        help='Input file name, containing the puzzle as xml')
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = get_args()
    PUZZLE = read_puzzle(ARGS.filename)
    RECS = build_records(PUZZLE["grid"], PUZZLE["horiz_clues"],
                         PUZZLE["vert_clues"])
    write_as_jsx(RECS, PUZZLE["title"], PUZZLE["info"])
