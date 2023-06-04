# author: Michal Sova
# year: 2021
# file: player.py
# description: File contains function for handling player input for movement (w, a, s, d)

def move(apos, size):
    npos = [-1, -1]    # [row, col]
    while True:
        line = input()
        if len(line) != 0:
            c = line[0]
            if c == 'w':
                npos[0] = apos[0] - 1
                npos[1] = apos[1]
            elif c == 'a':
                npos[0] = apos[0]
                npos[1] = apos[1] - 1
            elif c == 's':
                npos[0] = apos[0] + 1
                npos[1] = apos[1]
            elif c == 'd':
                npos[0] = apos[0]
                npos[1] = apos[1] + 1

            if 0 <= npos[0] < size and 0 <= npos[1] < size:
                return npos
            else:
                npos = [-1, -1]
