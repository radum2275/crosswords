#/usr/bin/python
#coding=utf-8

import sys
import re
import xml.etree.ElementTree as et

def writeHeader():
    print("\\documentclass[a4paper]{article}")
    print("\\pagestyle{empty}")
    #print("\\usepackage[margin=0.8in]{geometry}")
    print("\\usepackage{tikz}")
    print("\\usepackage{tabularx,colortbl}")
    print("\\usepackage{times}")
    print("\\usepackage{latexsym}")
    print("\\usepackage{color}")
    print("\\usepackage{url}")
    print("\\usepackage{ucs}")
    print("\\usepackage[utf8]{inputenc}")
    print("\usepackage{hyphsubst}")

    print("\\usepackage[T1]{fontenc}")
    print("\\usepackage[utf8]{inputenc}")

    print("\\usepackage[romanian]{babel}")
    print("\\usepackage{combelow}")
    print("\\usepackage{newunicodechar}")

    print("\\newunicodechar{Ș}{\cb{S}}")
    print("\\newunicodechar{ș}{\cb{s}}")
    print("\\newunicodechar{Ț}{\cb{T}}")
    print("\\newunicodechar{ț}{\cb{t}}")


    print("\\usepackage{fancyhdr}")
    print("\\pagestyle{fancy}")
    print("\\fancyhead{}")
    print("\\fancyfoot{}")
    print("\\lfoot{Rebus Online \\url{www.rebusonline.com}}")
    #print("\\lfoot{Rebus Online}")
    print("\\cfoot{}")
    #print("\\renewcommand{\\headheight}{4pt}")
    print("\\renewcommand{\\headrulewidth}{0.0pt}")
    print("\\renewcommand{\\footrulewidth}{0.4pt}")

    print("\\Large")
    print("\\normalsize")
    print("%don't want date printed")
    print("\\date{}")
    print("\\def\\lorel{Lorel}")
    print("\\begin{document}")

def writeCell(col, row, nrrows, size, color):
    startx = (col) * size;
    starty = (nrrows - row - 1) * size;
    endx = startx + size;
    endy = starty + size;
    print("\\fill [" + color + "] ("
            + str(startx) + "," + str(starty) + ") rectangle ("
            + str(endx) + "," + str(endy) + ");")
    print("\\draw [black] ("
            + str(startx) + "," + str(starty) + ") rectangle ("
            + str(endx) + "," + str(endy) + ");")

def writeGrid(grid, nrcols, nrrows, size):
    rows = grid.split("\n")
    print("\\begin{tikzpicture}\n");
    for row in range(0, nrrows):
        for col in range(0, nrcols):
            color = "white"
            if (rows[row][col] == '.'):
                color = "black"
            if ((rows[row][col]).islower()):
                color = "yellow"
            writeCell(col, row, nrrows, size, color)
    for col in range(0, nrcols):
        startx = col*size + 0.2;
        starty = (nrrows - 1)*size + 0.6;
        label = col + 1;
        print("\\draw (" + str(startx) + "," + str(starty) + ") node{" + str(label) + "};")
    for row in range(0, nrrows - 1):
        startx = 0.2
        starty = row*size + 0.6
        label = nrrows - row
        print("\\draw (" + str(startx) + "," + str(starty) + ") node{" + str(label) + "};")
    print("\\end{tikzpicture}\n")

def writeTitle(title):
    print("\\section*{" + title + "}")

def writeClues(grid, nrcols, nrrows, horiz, vert):
    print ("{\\bf ORIZONTAL:} ")
    rows = grid.split("\n")
    hclues = re.split(" - | – | - | - ", horiz)
    hci = 0
    for row in range(0, nrrows):
        print ("{\\bf " + str(row+1) + ")} ")
        slots = rows[row].split(".")
        nrslots = 0
        for s in slots:
            if len(s) > 1:
                nrslots = nrslots + 1
        for clueidx in range(0, nrslots):
            sys.stdout.write(hclues[hci+clueidx])
            if clueidx < nrslots - 1:
                print (" -- ")
            else:
                c = hclues[hci+clueidx][-1]
                l = ['!', '?', '.', '\)']
                if c not in l:
                    print (". ")
                else:
                    print ("")
        hci = hci + nrslots
    print ("\\\\\\\\")
    print ("{\\bf VERTICAL: } ")
    vclues = re.split(" - | – | - | - ", vert)
    vci = 0
    for col in range(0, nrcols):
        stringcol = ""
        print ("{\\bf " + str(col+1) + ")} ")
        for row in range(0, nrrows):
            stringcol = stringcol + rows[row][col]
        slots = stringcol.split(".")
        nrslots = 0
        for s in slots:
            if len(s) > 1:
                nrslots = nrslots + 1
        for clueidx in range(0, nrslots):
            sys.stdout.write(vclues[vci+clueidx])
            if clueidx < nrslots - 1:
                print (" -- ")
            else:
                c = vclues[vci+clueidx][-1]
                l = ['!', '?', '.', '\)', '\…']
                if c not in l:
                    print (". ")
                else:
                    print ("")
        vci = vci + nrslots

def writeOtherInfo(info):
    print ("\\\\\\\\" + info.replace("\n", "\\\\"))

def writeFooter():
    print("\\end{document}\n\\\\\n")


def main():
    with open(sys.argv[1], 'r') as file:
        tree = et.fromstring(file.read())
    for g in tree.iterfind('grid'):
        grid = g.text.strip()
    for t in tree.iterfind('title'):
        title = t.text.strip().encode("utf-8")
    for i in tree.iterfind('info_to_display'):
        info = i.text.strip().encode("utf-8")
    for hc in tree.iterfind('horizontal_clues'):
        horiz = hc.text.strip().encode("utf-8")
    for vc in tree.iterfind('vertical_clues'):
        vert = vc.text.strip().encode("utf-8")
    writeHeader()
    rows = grid.split("\n");
    nrrows = len(rows)
    nrcols = len(rows[0])
    writeGrid(grid, nrcols, nrrows, .8)
    writeTitle(str(title))
    writeClues(grid, nrcols, nrrows, horiz, vert)
    writeOtherInfo(info)
    writeFooter()

main()
